"""Immutable provenance and drain boundaries for DOJO paper rooms.

The live quote process is deliberately unable to reach an order API.  This
module adds the other half of the evidence boundary: an exact runtime/config
manifest for a paper cohort and a restartable, no-new-entry drain contract for
unresolved virtual exposure.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


SESSION_CONTRACT = "QR_VIRTUAL_MARKET_SESSION_V2"
DRAIN_CONTRACT = "QR_DOJO_PAPER_DRAIN_V1"
ZERO_SHA256 = "0" * 64


class DojoPaperContractError(ValueError):
    """The requested paper evidence boundary is incomplete or inconsistent."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_embedded_sha256(
    payload: dict[str, Any], *, digest_field: str, label: str
) -> None:
    digest = payload.get(digest_field)
    if not isinstance(digest, str) or len(digest) != 64:
        raise DojoPaperContractError(f"{label} has no valid {digest_field}")
    body = {key: value for key, value in payload.items() if key != digest_field}
    if canonical_sha256(body) != digest:
        raise DojoPaperContractError(f"{label} content hash mismatch")


def _relative_or_absolute(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def runtime_manifest(
    *,
    repo_root: Path,
    runner_path: Path,
    bot_module_path: Path | None,
    extra_paths: Iterable[Path] = (),
) -> dict[str, Any]:
    candidates = [
        runner_path,
        repo_root / "src/quant_rabbit/virtual_broker.py",
        repo_root / "src/quant_rabbit/dojo_paper_contract.py",
        repo_root / "src/quant_rabbit/analysis/market_status.py",
        repo_root / "src/quant_rabbit/broker/oanda.py",
    ]
    if bot_module_path is not None:
        candidates.append(bot_module_path)
        if bot_module_path.name == "combo_bot.py":
            candidates.append(bot_module_path.with_name("lab_bot.py"))
    candidates.extend(extra_paths)

    records = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if not resolved.is_file():
            raise DojoPaperContractError(
                f"runtime dependency is missing: {resolved}"
            )
        records.append(
            {
                "path": _relative_or_absolute(resolved, repo_root),
                "sha256": file_sha256(resolved),
                "size_bytes": resolved.stat().st_size,
            }
        )
    records.sort(key=lambda row: row["path"])
    payload = {"files": records}
    return {**payload, "manifest_sha256": canonical_sha256(payload)}


def _load_json_env(name: str) -> Any:
    raw = os.environ.get(name)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DojoPaperContractError(f"{name} is not valid JSON") from exc


def bot_contract(
    *,
    repo_root: Path,
    built_in_bot: str | None,
    bot_module: str | None,
    bot_config_env: str | None,
) -> dict[str, Any]:
    module_path: Path | None = None
    class_name: str | None = None
    if bot_module:
        raw_path, _, raw_class = bot_module.partition(":")
        module_path = Path(raw_path)
        if not module_path.is_absolute():
            module_path = repo_root / module_path
        module_path = module_path.resolve()
        class_name = raw_class or "Bot"

    inferred_env = bot_config_env
    if inferred_env is None and module_path is not None:
        if module_path.name == "combo_bot.py":
            inferred_env = "DOJO_BOT_COMBO"
        elif module_path.name == "lab_bot.py":
            inferred_env = "DOJO_BOT_CONFIG"
    config = _load_json_env(inferred_env) if inferred_env else None
    config_sha = canonical_sha256(config) if config is not None else None

    strategy_tags: list[str] = []
    configs: Iterable[Any]
    if isinstance(config, list):
        configs = config
    elif isinstance(config, dict):
        configs = [config]
    else:
        configs = []
    for row in configs:
        if not isinstance(row, dict):
            raise DojoPaperContractError("bot config entries must be objects")
        tag = row.get("strategy_tag") or row.get("strategy_id")
        if tag is not None:
            strategy_tags.append(str(tag))

    payload = {
        "kind": "module" if module_path is not None else (
            "built_in" if built_in_bot else "none"
        ),
        "built_in_name": built_in_bot,
        "module": (
            _relative_or_absolute(module_path, repo_root)
            if module_path is not None
            else None
        ),
        "module_sha256": file_sha256(module_path) if module_path else None,
        "class": class_name,
        "config_env": inferred_env,
        "config": config,
        "config_sha256": config_sha,
        "strategy_tags": strategy_tags,
    }
    return {**payload, "bot_contract_sha256": canonical_sha256(payload)}


def build_session_contract(
    *,
    experiment_id: str,
    room_id: str,
    candidate_id: str,
    room_kind: str,
    proof_mode: str,
    feed: str,
    pairs: list[str],
    initial_balance_jpy: float,
    slippage_pips: float,
    financing_pips_per_day: float,
    leverage: float,
    costs_explicit: bool,
    runtime: dict[str, Any],
    bot: dict[str, Any],
    source: dict[str, Any],
) -> dict[str, Any]:
    if proof_mode not in {"diagnostic", "formal"}:
        raise DojoPaperContractError(f"unsupported proof mode: {proof_mode}")
    if room_kind not in {"diagnostic", "single_strategy", "integrated", "ai"}:
        raise DojoPaperContractError(f"unsupported room kind: {room_kind}")
    if (
        not pairs
        or len(pairs) != len(set(pairs))
        or not all(isinstance(pair, str) and pair for pair in pairs)
    ):
        raise DojoPaperContractError("pairs must be non-empty and unique")
    numeric_values = (
        initial_balance_jpy,
        slippage_pips,
        financing_pips_per_day,
        leverage,
    )
    if not all(math.isfinite(float(value)) for value in numeric_values):
        raise DojoPaperContractError("balance, costs, and leverage must be finite")
    if (
        initial_balance_jpy <= 0
        or slippage_pips < 0
        or financing_pips_per_day < 0
        or leverage <= 0
    ):
        raise DojoPaperContractError("costs and leverage must be non-negative/positive")

    tags = list(bot.get("strategy_tags") or [])
    if len(tags) != len(set(tags)):
        raise DojoPaperContractError("strategy_tags must be unique")
    if proof_mode == "formal":
        if not costs_explicit:
            raise DojoPaperContractError(
                "formal paper requires explicit slippage, financing, and leverage"
            )
        for field_name, value in (
            ("experiment_id", experiment_id),
            ("room_id", room_id),
            ("candidate_id", candidate_id),
        ):
            if not value or value.startswith("diagnostic"):
                raise DojoPaperContractError(
                    f"formal paper requires a fixed {field_name}"
                )
        if room_kind == "single_strategy" and len(tags) != 1:
            raise DojoPaperContractError(
                "single-strategy formal paper requires exactly one strategy_tag"
            )
        if room_kind == "integrated" and len(tags) < 2:
            raise DojoPaperContractError(
                "integrated formal paper requires at least two strategy_tags"
            )
        if room_kind in {"single_strategy", "integrated"} and not bot.get(
            "config_sha256"
        ):
            raise DojoPaperContractError(
                "formal bot paper requires a content-addressed bot config"
            )
        if feed == "live":
            window_start = source.get("window_start_utc")
            window_end = source.get("window_end_utc")
            try:
                start_dt = datetime.fromisoformat(str(window_start))
                end_dt = datetime.fromisoformat(str(window_end))
            except ValueError as exc:
                raise DojoPaperContractError(
                    "formal live paper requires an ISO start/end window"
                ) from exc
            if start_dt.tzinfo is None or end_dt.tzinfo is None:
                raise DojoPaperContractError(
                    "formal live paper window must be timezone-aware"
                )
            if end_dt <= start_dt:
                raise DojoPaperContractError(
                    "formal live paper window end must follow its start"
                )
        elif feed == "replay":
            manifest_sha = source.get("source_manifest_sha256")
            if not isinstance(manifest_sha, str) or len(manifest_sha) != 64:
                raise DojoPaperContractError(
                    "formal replay requires a sealed source manifest sha256"
                )
        else:
            raise DojoPaperContractError(f"unsupported formal feed: {feed}")

    payload = {
        "contract": SESSION_CONTRACT,
        "schema_version": 2,
        "experiment_id": experiment_id,
        "room_id": room_id,
        "candidate_id": candidate_id,
        "room_kind": room_kind,
        "proof_mode": proof_mode,
        "proof_eligible": proof_mode == "formal",
        "feed": feed,
        "pairs": sorted(pairs),
        "initial_balance_jpy": float(initial_balance_jpy),
        "costs": {
            "slippage_pips_per_fill": float(slippage_pips),
            "financing_pips_per_day": float(financing_pips_per_day),
            "leverage": float(leverage),
            "explicit": costs_explicit,
        },
        "source": source,
        "bot": bot,
        "runtime": runtime,
        "authority": {
            "order_authority": "NONE",
            "broker_mutation_allowed": False,
            "live_permission": False,
        },
    }
    return {**payload, "session_contract_sha256": canonical_sha256(payload)}


def publish_immutable_json(path: Path, payload: dict[str, Any]) -> bool:
    """Publish once, or require byte-identical canonical content on retry."""

    expected = canonical_bytes(payload) + b"\n"
    if path.exists():
        if path.read_bytes() != expected:
            raise DojoPaperContractError(
                f"immutable contract conflicts with existing bytes: {path}"
            )
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("xb") as handle:
            handle.write(expected)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            if path.read_bytes() != expected:
                raise DojoPaperContractError(
                    f"immutable contract raced with conflicting bytes: {path}"
                )
            return False
        return True
    finally:
        tmp.unlink(missing_ok=True)


def read_ledger_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise DojoPaperContractError(f"ledger is missing: {path}")
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DojoPaperContractError(
                f"invalid ledger JSON at line {line_number}"
            ) from exc
        records.append(record)
    if not records:
        raise DojoPaperContractError("drain requires a non-empty source ledger")
    return records


def prepare_drain_contract(
    *,
    session_dir: Path,
    pairs: list[str],
    ceiling_minutes: int,
    allow_legacy_untagged: bool,
    slippage_pips: float,
    financing_pips_per_day: float,
    leverage: float,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    if isinstance(ceiling_minutes, bool) or ceiling_minutes <= 0:
        raise DojoPaperContractError("drain ceiling must be positive")
    if not pairs or len(pairs) != len(set(pairs)):
        raise DojoPaperContractError("drain pairs must be non-empty and unique")
    pending_inbox = sorted((session_dir / "inbox").glob("*.json"))
    if pending_inbox:
        raise DojoPaperContractError(
            "drain refuses pending agent actions: "
            + ",".join(path.name for path in pending_inbox)
        )

    ledger_path = session_dir / "ledger.jsonl"
    snapshot_path = session_dir / "broker_snapshot.json"
    if not snapshot_path.is_file():
        raise DojoPaperContractError("drain requires broker_snapshot.json")
    records = read_ledger_records(ledger_path)
    try:
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DojoPaperContractError("broker snapshot is invalid JSON") from exc
    latest = records[-1]
    if snapshot.get("ledger_sha") != latest.get("sha"):
        raise DojoPaperContractError(
            "drain snapshot does not match the ledger terminal sha"
        )

    contract_path = session_dir / "drain_contract.json"
    existing = None
    if contract_path.exists():
        try:
            existing = json.loads(contract_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise DojoPaperContractError("existing drain contract is invalid") from exc
        _verify_embedded_sha256(
            existing,
            digest_field="drain_contract_sha256",
            label="existing drain contract",
        )
        if existing.get("contract") != DRAIN_CONTRACT:
            raise DojoPaperContractError("existing drain contract type mismatch")

    if existing is None:
        if latest.get("event") != "SESSION_STOP":
            raise DojoPaperContractError(
                "first drain start requires a terminal SESSION_STOP"
            )
        source_terminal_sha = latest["sha"]
        source_snapshot_sha = canonical_sha256(snapshot)
    else:
        source_terminal_sha = existing.get("source_terminal_sha256")
        source_snapshot_sha = existing.get("source_snapshot_sha256")
        if not isinstance(source_snapshot_sha, str) or len(source_snapshot_sha) != 64:
            raise DojoPaperContractError(
                "existing drain contract has no valid source snapshot sha"
            )
        source_index = next(
            (
                idx
                for idx, record in enumerate(records)
                if record.get("sha") == source_terminal_sha
            ),
            None,
        )
        if source_index is None:
            raise DojoPaperContractError(
                "drain source terminal is no longer present in the ledger"
            )
        forbidden = [
            record.get("event")
            for record in records[source_index + 1 :]
            if record.get("event") == "SESSION_START"
        ]
        if forbidden:
            raise DojoPaperContractError(
                "normal strategy session appeared after the drain boundary"
            )

    positions = list(snapshot.get("positions") or [])
    orders = list(snapshot.get("orders") or [])
    instrument_set = {
        str(row.get("pair")) for row in positions + orders if row.get("pair")
    }
    if not instrument_set.issubset(set(pairs)):
        raise DojoPaperContractError(
            "drain pair list does not cover every unresolved instrument"
        )
    untagged = [row.get("trade_id") for row in positions if not row.get("strategy_tag")]
    if untagged and not allow_legacy_untagged:
        raise DojoPaperContractError(
            "drain refuses untagged positions without explicit legacy mode"
        )
    source_untagged = sorted(str(value) for value in untagged)
    if existing is not None:
        recorded_untagged = existing.get("legacy_untagged_trade_ids")
        if not isinstance(recorded_untagged, list) or not all(
            isinstance(value, str) for value in recorded_untagged
        ):
            raise DojoPaperContractError(
                "existing drain contract has no valid legacy trade inventory"
            )
        source_untagged = sorted(recorded_untagged)
        if not set(str(value) for value in untagged).issubset(source_untagged):
            raise DojoPaperContractError(
                "an untagged position appeared after the drain boundary"
            )

    source_session_contract = None
    session_contract_path = session_dir / "session_contract.json"
    if session_contract_path.exists():
        try:
            source_session_contract = json.loads(
                session_contract_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError as exc:
            raise DojoPaperContractError("source session contract is invalid") from exc
        _verify_embedded_sha256(
            source_session_contract,
            digest_field="session_contract_sha256",
            label="source session contract",
        )
        if source_session_contract.get("contract") != SESSION_CONTRACT:
            raise DojoPaperContractError("source session contract type mismatch")
        source_costs = source_session_contract.get("costs") or {}
        expected_costs = {
            "slippage_pips_per_fill": float(slippage_pips),
            "financing_pips_per_day": float(financing_pips_per_day),
            "leverage": float(leverage),
        }
        for key, value in expected_costs.items():
            if float(source_costs.get(key, float("nan"))) != value:
                raise DojoPaperContractError(
                    f"drain cost {key} differs from the source session"
                )

    payload = {
        "contract": DRAIN_CONTRACT,
        "schema_version": 1,
        "source_terminal_sha256": source_terminal_sha,
        "source_snapshot_sha256": source_snapshot_sha,
        "source_session_contract_sha256": (
            source_session_contract.get("session_contract_sha256")
            if source_session_contract
            else None
        ),
        "source_proof_eligible": bool(
            source_session_contract
            and source_session_contract.get("proof_eligible")
            and not source_untagged
        ),
        "proof_eligible": bool(
            source_session_contract
            and source_session_contract.get("proof_eligible")
            and not source_untagged
        ),
        "legacy_untagged_trade_ids": source_untagged,
        "pairs": sorted(pairs),
        "policy": {
            "new_entries_allowed": False,
            "pending_entry_orders_cancelled_at_start": True,
            "force_close_at_process_boundary": False,
            "position_ceiling_minutes": int(ceiling_minutes),
            "allow_legacy_untagged": bool(allow_legacy_untagged),
        },
        "costs": {
            "slippage_pips_per_fill": float(slippage_pips),
            "financing_pips_per_day": float(financing_pips_per_day),
            "leverage": float(leverage),
        },
        "runtime": runtime,
        "authority": {
            "order_authority": "NONE",
            "broker_mutation_allowed": False,
            "live_permission": False,
        },
    }
    payload = {**payload, "drain_contract_sha256": canonical_sha256(payload)}
    publish_immutable_json(contract_path, payload)
    return payload
