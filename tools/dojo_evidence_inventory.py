#!/usr/bin/env python3
"""Build a deterministic inventory of Git-external DOJO evidence.

Source paths must be project-relative, must remain inside one Git worktree, and
may contain only regular files and directories. Git-tracked files are inspected
for classification but omitted from the evidence manifest. By default the full
manifest is written to stdout. ``--output`` creates one new file without ever
overwriting an existing artifact.

An optional mirror comparison is deliberately stricter: it accepts exactly one
selected source directory and proves that a mirror directory has the identical
regular-file path set, sizes, and SHA-256 values, with no symlinks or special
files on either side.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Final, TypeAlias


SCHEMA_VERSION: Final = "DOJO_EVIDENCE_INVENTORY_V1"
CHUNK_BYTES: Final = 1024 * 1024
HashReader = Callable[[BinaryIO, int], tuple[str, int]]
StableFields: TypeAlias = tuple[int, int, int, int, int, int]


class InventoryError(RuntimeError):
    """Raised when evidence cannot be inventoried without ambiguity."""


def _git(root: Path, *args: str) -> bytes:
    environment = os.environ.copy()
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    result = subprocess.run(
        ["git", "-C", os.fspath(root), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=environment,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise InventoryError(
            f"git {' '.join(args)} failed: {detail or result.returncode}"
        )
    return result.stdout


def _worktree_root(project_root: Path) -> Path:
    candidate = project_root.expanduser().resolve(strict=True)
    raw = _git(candidate, "rev-parse", "--show-toplevel")
    root = Path(raw.decode("utf-8").strip()).resolve(strict=True)
    if candidate != root:
        raise InventoryError(
            f"project root must be the Git worktree root: {candidate} != {root}"
        )
    return root


def _git_identity(root: Path) -> dict[str, str]:
    head = _git(root, "rev-parse", "--verify", "HEAD").decode("ascii").strip()
    environment = os.environ.copy()
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    branch_result = subprocess.run(
        ["git", "-C", os.fspath(root), "symbolic-ref", "--quiet", "--short", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=environment,
    )
    if branch_result.returncode != 0:
        raise InventoryError("source worktree must have an attached branch")
    branch = branch_result.stdout.decode("utf-8").strip()
    if not head or not branch:
        raise InventoryError("source worktree HEAD and branch are required")
    return {"head": head, "branch": branch}


def _tracked_paths(root: Path) -> set[str]:
    raw = _git(root, "ls-files", "-z")
    return {
        item.decode("utf-8", errors="surrogateescape")
        for item in raw.split(b"\0")
        if item
    }


def _normalize_source_path(
    root: Path, raw_path: str | os.PathLike[str]
) -> tuple[str, Path]:
    raw = os.fspath(raw_path)
    if not raw or "\x00" in raw:
        raise InventoryError("source path must be a non-empty project-relative path")
    lexical = PurePosixPath(raw.replace(os.sep, "/"))
    if lexical.is_absolute() or any(part == ".." for part in lexical.parts):
        raise InventoryError(
            f"source path must be normalized and project-relative: {raw}"
        )
    normalized = lexical.as_posix()
    if normalized == ".":
        candidate = root
    else:
        candidate = root.joinpath(*lexical.parts)
    if lexical.parts and lexical.parts[0] == ".git":
        raise InventoryError(".git is outside the evidence inventory scope")
    try:
        resolved = candidate.resolve(strict=True)
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        raise InventoryError(f"source path is unavailable: {raw}: {exc}") from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise InventoryError(f"source path escapes the project root: {raw}") from exc
    _reject_symlink_components(root, candidate)
    return normalized, resolved


def _reject_symlink_components(root: Path, candidate: Path) -> None:
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise InventoryError(f"path escapes the project root: {candidate}") from exc
    current = root
    for part in relative.parts:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except OSError as exc:
            raise InventoryError(
                f"cannot inspect path component: {current}: {exc}"
            ) from exc
        if stat.S_ISLNK(mode):
            raise InventoryError(
                f"symlink is forbidden: {current.relative_to(root).as_posix()}"
            )


def _walk_files(
    root: Path, selected_path: Path, *, skip_git_metadata: bool = True
) -> Iterable[Path]:
    try:
        selected_mode = selected_path.lstat().st_mode
    except OSError as exc:
        raise InventoryError(
            f"cannot inspect selected path: {selected_path}: {exc}"
        ) from exc
    if stat.S_ISLNK(selected_mode):
        raise InventoryError(f"symlink is forbidden: {selected_path.relative_to(root)}")
    if stat.S_ISREG(selected_mode):
        yield selected_path
        return
    if not stat.S_ISDIR(selected_mode):
        raise InventoryError(
            f"non-regular source is forbidden: {selected_path.relative_to(root)}"
        )

    stack = [selected_path]
    while stack:
        directory = stack.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as exc:
            raise InventoryError(f"cannot scan directory: {directory}: {exc}") from exc
        child_directories: list[Path] = []
        for entry in entries:
            path = Path(entry.path)
            relative = path.relative_to(root).as_posix()
            if skip_git_metadata and (
                relative == ".git" or relative.startswith(".git/")
            ):
                continue
            try:
                if entry.is_symlink():
                    raise InventoryError(f"symlink is forbidden: {relative}")
                if entry.is_file(follow_symlinks=False):
                    yield path
                elif entry.is_dir(follow_symlinks=False):
                    child_directories.append(path)
                else:
                    raise InventoryError(f"non-regular source is forbidden: {relative}")
            except OSError as exc:
                raise InventoryError(
                    f"cannot inspect source: {relative}: {exc}"
                ) from exc
        stack.extend(reversed(child_directories))


def _hash_open_file(handle: BinaryIO, expected_size: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    observed_bytes = 0
    while True:
        chunk = handle.read(CHUNK_BYTES)
        if not chunk:
            break
        digest.update(chunk)
        observed_bytes += len(chunk)
        if observed_bytes > expected_size:
            raise InventoryError("file grew while it was being hashed")
    return digest.hexdigest(), observed_bytes


def _stable_fields(value: os.stat_result) -> StableFields:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _open_file_beneath_root(root: Path, path: Path) -> int:
    """Open one file through no-follow directory descriptors rooted in the worktree."""

    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise InventoryError(f"source path escapes the project root: {path}") from exc
    if not relative.parts:
        raise InventoryError("the project root is not a regular evidence file")

    directory_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    directory_flags |= getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    file_flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptors: list[int] = []
    try:
        current = os.open(root, directory_flags)
        descriptors.append(current)
        for component in relative.parts[:-1]:
            current = os.open(component, directory_flags, dir_fd=current)
            descriptors.append(current)
        return os.open(relative.parts[-1], file_flags, dir_fd=current)
    except OSError as exc:
        relative_text = relative.as_posix()
        raise InventoryError(
            f"cannot open source file beneath the project root: {relative_text}: {exc}"
        ) from exc
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _hash_file_stably(
    *,
    root: Path,
    path: Path,
    hash_reader: HashReader,
) -> tuple[str, int, StableFields]:
    relative = path.relative_to(root).as_posix()
    _reject_symlink_components(root, path)
    try:
        before_path = path.lstat()
    except OSError as exc:
        raise InventoryError(f"cannot stat source file: {relative}: {exc}") from exc
    if not stat.S_ISREG(before_path.st_mode):
        raise InventoryError(f"source is no longer a regular file: {relative}")

    descriptor = _open_file_beneath_root(root, path)

    try:
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            before_fd = os.fstat(handle.fileno())
            if _stable_fields(before_fd) != _stable_fields(before_path):
                raise InventoryError(f"source changed before hashing: {relative}")
            digest, observed_bytes = hash_reader(handle, before_fd.st_size)
            after_fd = os.fstat(handle.fileno())
    except InventoryError:
        raise
    except OSError as exc:
        raise InventoryError(f"cannot read source file: {relative}: {exc}") from exc

    try:
        after_path = path.lstat()
    except OSError as exc:
        raise InventoryError(
            f"source changed after hashing: {relative}: {exc}"
        ) from exc
    if (
        _stable_fields(before_fd) != _stable_fields(after_fd)
        or _stable_fields(before_path) != _stable_fields(after_path)
        or observed_bytes != before_fd.st_size
    ):
        raise InventoryError(f"source changed while hashing: {relative}")
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise InventoryError(f"hash reader returned an invalid SHA-256: {relative}")
    _reject_symlink_components(root, path)
    return digest, before_fd.st_size, _stable_fields(after_path)


def _resolve_mirror_root(raw_root: Path) -> Path:
    """Resolve a real mirror directory while rejecting a symlink root."""

    raw = raw_root.expanduser()
    try:
        raw_state = raw.lstat()
    except OSError as exc:
        raise InventoryError(f"mirror root is unavailable: {raw}: {exc}") from exc
    if stat.S_ISLNK(raw_state.st_mode):
        raise InventoryError(f"mirror root symlink is forbidden: {raw}")
    if not stat.S_ISDIR(raw_state.st_mode):
        raise InventoryError(f"mirror root must be a directory: {raw}")
    try:
        resolved = raw.resolve(strict=True)
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        raise InventoryError(f"mirror root is unavailable: {raw}: {exc}") from exc
    try:
        resolved_state = resolved.lstat()
    except OSError as exc:
        raise InventoryError(f"cannot inspect mirror root: {resolved}: {exc}") from exc
    if not stat.S_ISDIR(resolved_state.st_mode):
        raise InventoryError(f"mirror root must be a directory: {resolved}")
    return resolved


def _candidate_paths(
    *,
    root: Path,
    selected_paths: Iterable[Path],
    tracked: set[str] | None,
) -> dict[str, Path]:
    """Collect a complete regular-file set, optionally excluding tracked paths."""

    candidates: dict[str, Path] = {}
    for selected in selected_paths:
        for path in _walk_files(
            root,
            selected,
            skip_git_metadata=tracked is not None,
        ):
            relative = path.relative_to(root).as_posix()
            if tracked is None or relative not in tracked:
                candidates[relative] = path
    return candidates


def _assert_file_states_unchanged(
    *,
    root: Path,
    candidates: Mapping[str, Path],
    states: Mapping[str, StableFields],
    label: str,
) -> None:
    for relative in sorted(candidates):
        path = candidates[relative]
        _reject_symlink_components(root, path)
        try:
            current = path.lstat()
        except OSError as exc:
            raise InventoryError(
                f"{label} changed after hashing: {relative}: {exc}"
            ) from exc
        if (
            not stat.S_ISREG(current.st_mode)
            or _stable_fields(current) != states[relative]
        ):
            raise InventoryError(f"{label} changed after hashing: {relative}")


def _bounded_paths(values: Iterable[str]) -> list[str]:
    return sorted(values)[:20]


def _inventory_digest(files: list[dict[str, object]]) -> str:
    total_bytes = sum(int(row["size"]) for row in files)
    body = {
        "file_count": len(files),
        "total_bytes": total_bytes,
        "files": files,
    }
    return hashlib.sha256(_canonical_json_bytes(body)).hexdigest()


def _path_is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _output_target(
    *,
    raw_output: Path,
    project_root: Path,
    source_paths: Sequence[str | os.PathLike[str]],
    mirror_root: Path | None,
) -> Path:
    """Resolve and validate a not-yet-created output outside every scan scope."""

    try:
        parent = raw_output.expanduser().parent.resolve(strict=True)
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        raise InventoryError(
            f"output parent must already exist: {raw_output}: {exc}"
        ) from exc
    output = parent / raw_output.name
    try:
        output.lstat()
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise InventoryError(f"cannot inspect output path: {output}: {exc}") from exc
    else:
        raise InventoryError(f"output already exists: {output}")

    root = _worktree_root(project_root)
    selected: list[Path] = []
    for raw_path in source_paths:
        _, resolved = _normalize_source_path(root, raw_path)
        selected.append(resolved)
    for scope in selected:
        if _path_is_within(output, scope):
            raise InventoryError(
                f"output must be outside scanned source scope: {scope}"
            )
    if mirror_root is not None:
        mirror = _resolve_mirror_root(mirror_root)
        if _path_is_within(output, mirror):
            raise InventoryError(f"output must be outside mirror scope: {mirror}")
    return output


def _write_json_create_new(path: Path, value: object) -> None:
    """Durably create one JSON file using O_EXCL; never replace old evidence."""

    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    directory_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    directory_flags |= getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    file_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    file_flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    parent_descriptor = os.open(path.parent, directory_flags)
    created = False
    try:
        try:
            descriptor = os.open(path.name, file_flags, 0o644, dir_fd=parent_descriptor)
        except FileExistsError as exc:
            raise InventoryError(f"output already exists: {path}") from exc
        created = True
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.fsync(parent_descriptor)
        except BaseException:
            try:
                os.unlink(path.name, dir_fd=parent_descriptor)
            except OSError:
                pass
            raise
    finally:
        os.close(parent_descriptor)
    if not created:  # pragma: no cover - defensive invariant
        raise InventoryError(f"output was not created: {path}")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def build_manifest(
    *,
    project_root: Path,
    source_paths: Sequence[str | os.PathLike[str]],
    mirror_root: Path | None = None,
    _hash_reader: HashReader = _hash_open_file,
    _mirror_hash_reader: HashReader = _hash_open_file,
) -> dict[str, object]:
    """Return a deterministic source inventory and optional mirror receipt."""

    if not source_paths:
        raise InventoryError("at least one source path is required")
    if mirror_root is not None and len(source_paths) != 1:
        raise InventoryError(
            "mirror verification requires exactly one source directory"
        )
    root = _worktree_root(project_root)
    identity_before = _git_identity(root)
    tracked = _tracked_paths(root)

    normalized_sources: dict[str, Path] = {}
    for raw_path in source_paths:
        relative, resolved = _normalize_source_path(root, raw_path)
        normalized_sources[relative] = resolved
    if mirror_root is not None and len(normalized_sources) != 1:
        raise InventoryError(
            "mirror verification requires exactly one source directory"
        )

    selected_states = {
        relative: _stable_fields(path.lstat())
        for relative, path in normalized_sources.items()
    }
    candidates = _candidate_paths(
        root=root,
        selected_paths=normalized_sources.values(),
        tracked=tracked,
    )

    files: list[dict[str, object]] = []
    source_states: dict[str, StableFields] = {}
    total_bytes = 0
    for relative in sorted(candidates):
        digest, size, stable_state = _hash_file_stably(
            root=root,
            path=candidates[relative],
            hash_reader=_hash_reader,
        )
        files.append({"path": relative, "size": size, "sha256": digest})
        source_states[relative] = stable_state
        total_bytes += size

    mirror_comparison: dict[str, object] | None = None
    mirror: Path | None = None
    mirror_candidates: dict[str, Path] = {}
    mirror_states: dict[str, StableFields] = {}
    mirror_root_state: StableFields | None = None
    if mirror_root is not None:
        selected_relative, selected_directory = next(iter(normalized_sources.items()))
        selected_mode = selected_directory.lstat().st_mode
        if not stat.S_ISDIR(selected_mode):
            raise InventoryError(
                "mirror verification requires exactly one source directory"
            )
        all_source_candidates = _candidate_paths(
            root=root,
            selected_paths=[selected_directory],
            tracked=None,
        )
        tracked_in_scope = set(all_source_candidates) - set(candidates)
        if tracked_in_scope:
            raise InventoryError(
                "mirror verification source must contain only Git-external files: "
                f"tracked={_bounded_paths(tracked_in_scope)}"
            )
        mirror = _resolve_mirror_root(mirror_root)
        mirror_root_state = _stable_fields(mirror.lstat())
        mirror_candidates = _candidate_paths(
            root=mirror,
            selected_paths=[mirror],
            tracked=None,
        )

        source_by_relative = {
            path.relative_to(selected_directory).as_posix(): path
            for path in candidates.values()
        }
        source_paths_relative = set(source_by_relative)
        mirror_paths_relative = set(mirror_candidates)
        missing = source_paths_relative - mirror_paths_relative
        extra = mirror_paths_relative - source_paths_relative
        if missing or extra:
            raise InventoryError(
                "mirror file set mismatch: "
                f"missing={_bounded_paths(missing)}, extra={_bounded_paths(extra)}"
            )

        source_rows_by_relative = {
            candidates[str(row["path"])].relative_to(selected_directory).as_posix(): {
                "path": candidates[str(row["path"])]
                .relative_to(selected_directory)
                .as_posix(),
                "size": row["size"],
                "sha256": row["sha256"],
            }
            for row in files
        }
        mirror_rows: list[dict[str, object]] = []
        for relative in sorted(mirror_candidates):
            expected = source_rows_by_relative[relative]
            try:
                mirror_size = mirror_candidates[relative].lstat().st_size
            except OSError as exc:
                raise InventoryError(
                    f"cannot inspect mirror source: {relative}: {exc}"
                ) from exc
            if mirror_size != expected["size"]:
                raise InventoryError(
                    "mirror size mismatch: "
                    f"{relative}: source={expected['size']}, mirror={mirror_size}"
                )
            digest, size, stable_state = _hash_file_stably(
                root=mirror,
                path=mirror_candidates[relative],
                hash_reader=_mirror_hash_reader,
            )
            source_state_key = source_by_relative[relative].relative_to(root).as_posix()
            source_state = source_states[source_state_key]
            if stable_state[:2] == source_state[:2]:
                raise InventoryError(
                    "mirror is a hardlink alias of the source: " + relative
                )
            if digest != expected["sha256"]:
                raise InventoryError(f"mirror SHA-256 mismatch: {relative}")
            mirror_rows.append({"path": relative, "size": size, "sha256": digest})
            mirror_states[relative] = stable_state

        source_relative_rows = [
            source_rows_by_relative[relative]
            for relative in sorted(source_rows_by_relative)
        ]
        source_inventory_sha256 = _inventory_digest(source_relative_rows)
        mirror_inventory_sha256 = _inventory_digest(mirror_rows)
        if source_inventory_sha256 != mirror_inventory_sha256:
            raise InventoryError("mirror inventory digest mismatch")
        receipt_body: dict[str, object] = {
            "status": "VERIFIED_EXACT",
            "source_selected_path": selected_relative,
            "mirror_root": os.fspath(mirror),
            "file_count": len(source_relative_rows),
            "total_bytes": total_bytes,
            "source_inventory_sha256": source_inventory_sha256,
            "mirror_inventory_sha256": mirror_inventory_sha256,
        }
        mirror_comparison = {
            **receipt_body,
            "receipt_sha256": hashlib.sha256(
                _canonical_json_bytes(receipt_body)
            ).hexdigest(),
        }

    tracked_after = _tracked_paths(root)
    if tracked_after != tracked:
        raise InventoryError("Git tracked path set changed during inventory")
    identity_after = _git_identity(root)
    if identity_after != identity_before:
        raise InventoryError("source worktree HEAD or branch changed during inventory")

    candidates_after = _candidate_paths(
        root=root,
        selected_paths=normalized_sources.values(),
        tracked=tracked,
    )
    if set(candidates_after) != set(candidates):
        raise InventoryError("source file set changed during inventory")
    _assert_file_states_unchanged(
        root=root,
        candidates=candidates,
        states=source_states,
        label="source",
    )
    for relative, selected in normalized_sources.items():
        try:
            selected_after = selected.lstat()
        except OSError as exc:
            raise InventoryError(
                f"selected source changed during inventory: {relative}: {exc}"
            ) from exc
        if _stable_fields(selected_after) != selected_states[relative]:
            raise InventoryError(
                f"selected source changed during inventory: {relative}"
            )

    if mirror is not None:
        mirror_candidates_after = _candidate_paths(
            root=mirror,
            selected_paths=[mirror],
            tracked=None,
        )
        if set(mirror_candidates_after) != set(mirror_candidates):
            raise InventoryError("mirror file set changed during inventory")
        _assert_file_states_unchanged(
            root=mirror,
            candidates=mirror_candidates,
            states=mirror_states,
            label="mirror",
        )
        try:
            mirror_after = mirror.lstat()
        except OSError as exc:
            raise InventoryError(
                f"mirror root changed during inventory: {exc}"
            ) from exc
        if _stable_fields(mirror_after) != mirror_root_state:
            raise InventoryError("mirror root changed during inventory")

    body: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "source_worktree": identity_before,
        "selected_paths": sorted(normalized_sources),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "files": files,
    }
    if mirror_comparison is not None:
        body["mirror_comparison"] = mirror_comparison
    return {
        **body,
        "manifest_sha256": hashlib.sha256(_canonical_json_bytes(body)).hexdigest(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inventory Git-external DOJO evidence and optionally verify one exact mirror."
        )
    )
    parser.add_argument(
        "source_paths",
        nargs="+",
        metavar="PROJECT_RELATIVE_PATH",
        help="Regular file or directory inside the source worktree.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Git worktree root (default: current directory).",
    )
    parser.add_argument(
        "--mirror-root",
        type=Path,
        help="Exact mirror of one selected source directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Create a new manifest file; existing files are never overwritten.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        output_path = None
        if args.output is not None:
            output_path = _output_target(
                raw_output=args.output,
                project_root=args.project_root,
                source_paths=args.source_paths,
                mirror_root=args.mirror_root,
            )
        manifest = build_manifest(
            project_root=args.project_root,
            source_paths=args.source_paths,
            mirror_root=args.mirror_root,
        )
        if output_path is not None:
            output_after = _output_target(
                raw_output=args.output,
                project_root=args.project_root,
                source_paths=args.source_paths,
                mirror_root=args.mirror_root,
            )
            if output_after != output_path:
                raise InventoryError("output target changed during inventory")
            _write_json_create_new(output_path, manifest)
    except (InventoryError, OSError) as exc:
        print(json.dumps({"error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    if output_path is None:
        print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))
    else:
        print(
            json.dumps(
                {
                    "status": "CREATED",
                    "output": os.fspath(output_path),
                    "manifest_sha256": manifest["manifest_sha256"],
                    "file_count": manifest["file_count"],
                    "total_bytes": manifest["total_bytes"],
                    "mirror_status": (
                        manifest.get("mirror_comparison", {}).get("status")
                    ),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
