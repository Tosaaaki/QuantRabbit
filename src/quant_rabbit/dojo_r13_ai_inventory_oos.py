"""Causal AI inventory-overlay study for the immutable r13 January replay.

The r13 bot-only job is never executed again by this module.  Its economic
transcripts are read-only source evidence.  The module extracts the immutable
trade schedule and observed quote path, then applies inventory-only overlays to
that schedule in a separate paper account.

The experiment deliberately separates:

* ``A_BOT_ONLY``: immutable transcript-derived baseline.
* ``B_INVENTORY_ONLY``: inventory management without a forecast head.
* ``C_FORECAST_INVENTORY``: narrative + forecast + adaptive inventory.

Worker responses are accepted only against an exact point-in-time packet hash.
No broker client, live gateway, deployment hook, or order authority is imported
or exposed here.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import stat
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from quant_rabbit.dojo_portfolio_replay_reducer import canonical_portfolio_sha256


STUDY_CONTRACT: Final = "QR_DOJO_R13_AI_INVENTORY_OOS_STUDY_V1"
PREPARED_CONTRACT: Final = "QR_DOJO_R13_AI_INVENTORY_PREPARED_INPUT_V1"
PACKET_CONTRACT: Final = "QR_DOJO_R13_AI_INVENTORY_PACKET_V1"
RESPONSE_CONTRACT: Final = "QR_DOJO_R13_AI_INVENTORY_RESPONSE_V1"
CELL_CONTRACT: Final = "QR_DOJO_R13_AI_INVENTORY_CELL_V1"
SCHEMA_VERSION: Final = 1

A_BOT_ONLY: Final = "A_BOT_ONLY"
B_INVENTORY_ONLY: Final = "B_INVENTORY_ONLY"
C_FORECAST_INVENTORY: Final = "C_FORECAST_INVENTORY"
ARMS: Final = (A_BOT_ONLY, B_INVENTORY_ONLY, C_FORECAST_INVENTORY)

CADENCE_SECONDS: Final = {
    "FIXED_5M": 5 * 60,
    "FIXED_15M": 15 * 60,
    "FIXED_30M": 30 * 60,
    "FIXED_60M": 60 * 60,
    "FIXED_120M": 120 * 60,
    "EVENT_DRIVEN": None,
    "ADAPTIVE": None,
}
FORECAST_DIRECTIONS: Final = ("UP", "DOWN", "RANGE", "UNCERTAIN")
ACTIONS: Final = (
    "HOLD",
    "PAUSE_NEW_ENTRIES",
    "RESUME",
    "REDUCE_LONG",
    "REDUCE_SHORT",
    "PARTIAL_CLOSE",
    "CLOSE_RISKY",
    "CLOSE_ALL",
)
DIRECTION_RESTRICTIONS: Final = (
    "NONE",
    "LONG_ONLY",
    "SHORT_ONLY",
    "NO_NEW_LONGS",
    "NO_NEW_SHORTS",
)
PHASE_A_POLICY_PROFILES: Final = {
    "PROTECTIVE_V1": {
        "close_loss_progress": 0.58,
        "partial_loss_progress": 0.36,
        "giveback_trigger": 0.16,
        "tp_progress_trigger": 0.25,
    },
    "BALANCED_V1": {
        "close_loss_progress": 0.72,
        "partial_loss_progress": 0.48,
        "giveback_trigger": 0.30,
        "tp_progress_trigger": 0.35,
    },
    "PATIENT_V1": {
        "close_loss_progress": 0.86,
        "partial_loss_progress": 0.66,
        "giveback_trigger": 0.45,
        "tp_progress_trigger": 0.50,
    },
}

_PHASE_ORDER: Final = {"O": 0, "H": 1, "L": 2, "C": 3}
_AUTHORITY: Final = {
    "research_only": True,
    "paper_replay_only": True,
    "live_permission": False,
    "broker_mutation_allowed": False,
    "order_authority": "NONE",
    "automatic_deployment_allowed": False,
    "promotion_eligible": False,
}
_ZERO_SHA: Final = "0" * 64


class DojoR13AIInventoryError(ValueError):
    """The immutable input, point-in-time packet, or worker output is invalid."""


def _copy_json(value: Any) -> Any:
    try:
        return json.loads(
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    except (TypeError, ValueError) as exc:
        raise DojoR13AIInventoryError("value is not strict JSON") from exc


def _mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DojoR13AIInventoryError(f"{field_name} must be an object")
    return _copy_json(dict(value))


def _sequence(value: Any, field_name: str) -> list[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise DojoR13AIInventoryError(f"{field_name} must be an array")
    return list(value)


def _exact(value: Mapping[str, Any], keys: set[str], field_name: str) -> None:
    if set(value) != keys:
        raise DojoR13AIInventoryError(f"{field_name} schema mismatch")


def _identifier(value: Any, field_name: str, *, maximum: int = 240) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > maximum
        or any(ord(character) < 32 for character in value)
    ):
        raise DojoR13AIInventoryError(f"{field_name} is invalid")
    return value


def _sha256(value: Any, field_name: str, *, allow_zero: bool = False) -> str:
    digest = _identifier(value, field_name, maximum=64)
    if (
        len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or (not allow_zero and digest == _ZERO_SHA)
    ):
        raise DojoR13AIInventoryError(f"{field_name} must be lowercase SHA-256")
    return digest


def _integer(value: Any, field_name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DojoR13AIInventoryError(
            f"{field_name} must be an integer >= {minimum}"
        )
    return value


def _finite(
    value: Any,
    field_name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DojoR13AIInventoryError(f"{field_name} must be numeric")
    result = float(value)
    if (
        not math.isfinite(result)
        or (minimum is not None and result < minimum)
        or (maximum is not None and result > maximum)
    ):
        raise DojoR13AIInventoryError(f"{field_name} is outside its bounds")
    return result


def _strict_bool(value: Any, field_name: str) -> bool:
    if value.__class__ is not bool:
        raise DojoR13AIInventoryError(f"{field_name} must be boolean")
    return bool(value)


def _canonical_record_sha256(record: Mapping[str, Any]) -> str:
    body = {key: value for key, value in record.items() if key != "record_sha256"}
    return canonical_portfolio_sha256(body)


def _utc_iso(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def _file_sha256(path: Path) -> tuple[int, str]:
    target = path.resolve(strict=True)
    before = target.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
        raise DojoR13AIInventoryError(f"input is not a regular file: {target}")
    descriptor = os.open(
        target,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    digest = hashlib.sha256()
    size = 0
    try:
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
        opened = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = target.stat(follow_symlinks=False)
    identities = {
        (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
        for item in (before, opened, after)
    }
    if len(identities) != 1 or size != before.st_size:
        raise DojoR13AIInventoryError(f"input changed while hashing: {target}")
    return size, digest.hexdigest()


def _read_json(path: Path, *, maximum_bytes: int = 32 * 1024 * 1024) -> dict[str, Any]:
    target = path.resolve(strict=True)
    before = target.stat(follow_symlinks=False)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_size <= 0
        or before.st_size > maximum_bytes
    ):
        raise DojoR13AIInventoryError(f"JSON input is outside its bound: {target}")
    descriptor = os.open(
        target,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        payload = b""
        while chunk := os.read(descriptor, 1024 * 1024):
            payload += chunk
            if len(payload) > maximum_bytes:
                raise DojoR13AIInventoryError("JSON input exceeded its read bound")
        opened = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = target.stat(follow_symlinks=False)
    identities = {
        (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
        for item in (before, opened, after)
    }
    if len(identities) != 1 or len(payload) != before.st_size:
        raise DojoR13AIInventoryError("JSON input changed while reading")
    try:
        return _mapping(json.loads(payload), str(target))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise DojoR13AIInventoryError(f"invalid JSON input: {target}") from exc


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(temporary, flags, 0o600)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def _atomic_gzip_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
                compressed.write(
                    json.dumps(
                        value,
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                )
            raw.flush()
            os.fsync(raw.fileno())
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def _read_gzip_json(path: Path) -> Any:
    target = path.resolve(strict=True)
    if not stat.S_ISREG(target.stat(follow_symlinks=False).st_mode):
        raise DojoR13AIInventoryError("prepared input is not a regular file")
    try:
        with gzip.open(target, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise DojoR13AIInventoryError("prepared gzip JSON is invalid") from exc


def _quote_map(frame: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    return {
        str(row["pair"]): {
            "bid": float(row["bid"]),
            "ask": float(row["ask"]),
        }
        for row in frame["quotes"]
    }


def _quote_to_jpy(
    amount: float, currency: str, quotes: Mapping[str, Mapping[str, float]]
) -> float:
    if currency == "JPY":
        return amount
    pair = f"{currency}_JPY"
    quote = quotes.get(pair)
    if quote is None:
        raise DojoR13AIInventoryError(f"missing JPY conversion quote for {currency}")
    return amount * (float(quote["bid"]) + float(quote["ask"])) / 2.0


def _close_fill(
    trade: Mapping[str, Any],
    frame: Mapping[str, Any],
    *,
    exit_slippage_price: float,
    protected_tp: bool = False,
) -> tuple[float, float]:
    quotes = _quote_map(frame)
    pair_quote = quotes[trade["pair"]]
    if protected_tp:
        return float(trade["tp_price"]), 0.0
    if trade["side"] == "LONG":
        return float(pair_quote["bid"]) - exit_slippage_price, exit_slippage_price
    return float(pair_quote["ask"]) + exit_slippage_price, exit_slippage_price


def _trade_price_pnl_jpy(
    trade: Mapping[str, Any],
    units: float,
    fill_price: float,
    quotes: Mapping[str, Mapping[str, float]],
) -> float:
    delta = (
        fill_price - float(trade["entry_price"])
        if trade["side"] == "LONG"
        else float(trade["entry_price"]) - fill_price
    )
    quote_currency = str(trade["pair"]).split("_", 1)[1]
    return _quote_to_jpy(units * delta, quote_currency, quotes)


def _classify_baseline_close(
    trade: Mapping[str, Any], frame: Mapping[str, Any]
) -> str:
    quote = _quote_map(frame)[trade["pair"]]
    if trade["side"] == "LONG":
        if float(quote["bid"]) <= float(trade["sl_price"]):
            return "STOP_LOSS"
        if trade["tp_price"] is not None and float(quote["bid"]) >= float(
            trade["tp_price"]
        ):
            return "TAKE_PROFIT"
    else:
        if float(quote["ask"]) >= float(trade["sl_price"]):
            return "STOP_LOSS"
        if trade["tp_price"] is not None and float(quote["ask"]) <= float(
            trade["tp_price"]
        ):
            return "TAKE_PROFIT"
    if int(frame["epoch"]) >= int(trade["hard_exit_epoch"]):
        return "HARD_EXIT"
    return "WORKER_CLOSE"


def _transcript_snapshot_rows(
    path: Path,
    *,
    expected_file_sha256: str,
    coordinate_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], str]:
    """Read only HEADER/POST_EXIT_SNAPSHOT/TERMINAL from one immutable JSONL."""

    digest = hashlib.sha256()
    header: dict[str, Any] | None = None
    frames: list[dict[str, Any]] = []
    terminal: dict[str, Any] | None = None
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            digest.update(raw)
            if (
                b'"event_type":"HEADER"' not in raw
                and b'"event_type":"POST_EXIT_SNAPSHOT"' not in raw
                and b'"event_type":"TERMINAL_SUCCESS"' not in raw
            ):
                continue
            try:
                record = _mapping(json.loads(raw), f"transcript line {line_number}")
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise DojoR13AIInventoryError(
                    f"invalid transcript JSON at line {line_number}"
                ) from exc
            event_type = record.get("event_type")
            if event_type == "HEADER":
                header = _mapping(record["payload"], "transcript header")
                if header.get("coordinate_id") != coordinate_id:
                    raise DojoR13AIInventoryError("transcript coordinate mismatch")
            elif event_type == "POST_EXIT_SNAPSHOT":
                snapshot = _mapping(record["payload"]["snapshot"], "snapshot")
                frames.append(
                    {
                        "epoch": _integer(snapshot["epoch"], "snapshot.epoch"),
                        "phase": _identifier(snapshot["phase"], "snapshot.phase"),
                        "intrabar": _identifier(
                            snapshot["intrabar"], "snapshot.intrabar"
                        ),
                        "quote_watermark": _integer(
                            snapshot["quote_watermark"], "snapshot.quote_watermark"
                        ),
                        "quotes": snapshot["quotes"],
                        "account": snapshot["account"],
                        "positions": snapshot["positions"],
                        "pending_orders": snapshot["pending_orders"],
                        "snapshot_sha256": snapshot["snapshot_sha256"],
                    }
                )
            elif event_type == "TERMINAL_SUCCESS":
                terminal = _mapping(record["payload"], "terminal")
    actual = digest.hexdigest()
    if actual != expected_file_sha256:
        raise DojoR13AIInventoryError("immutable transcript file SHA mismatch")
    if header is None or terminal is None or not frames:
        raise DojoR13AIInventoryError("transcript lacks header/snapshots/terminal")
    return header, frames, terminal, actual


def _derive_trades(
    frames: Sequence[Mapping[str, Any]],
    *,
    cost_policy: Mapping[str, Any],
    terminal: Mapping[str, Any],
) -> list[dict[str, Any]]:
    slippage = {
        row["pair"]: float(row["exit_slippage_price"])
        for row in cost_policy["slippage_by_pair"]
    }
    financing = {
        row["pair"]: {
            "LONG": float(row["long_cost_jpy_per_unit_day"]),
            "SHORT": float(row["short_cost_jpy_per_unit_day"]),
        }
        for row in cost_policy["financing_by_pair"]
    }
    active: dict[str, dict[str, Any]] = {}
    completed: list[dict[str, Any]] = []
    for frame_index, frame in enumerate(frames):
        current = {row["position_id"]: _copy_json(row) for row in frame["positions"]}
        for position_id in sorted(set(active) - set(current)):
            trade = active.pop(position_id)
            close_reason = _classify_baseline_close(trade, frame)
            protected = close_reason == "TAKE_PROFIT"
            fill_price, exit_slip = _close_fill(
                trade,
                frame,
                exit_slippage_price=slippage[trade["pair"]],
                protected_tp=protected,
            )
            quotes = _quote_map(frame)
            gross = _trade_price_pnl_jpy(
                trade, float(trade["units"]), fill_price, quotes
            )
            elapsed = max(0, int(frame["epoch"]) - int(trade["opened_epoch"]))
            finance = (
                float(trade["units"])
                * financing[trade["pair"]][trade["side"]]
                * elapsed
                / 86400.0
            )
            completed.append(
                {
                    **trade,
                    "close_frame_index": frame_index,
                    "closed_epoch": int(frame["epoch"]),
                    "closed_phase": str(frame["phase"]),
                    "close_reason": close_reason,
                    "baseline_fill_price": fill_price,
                    "baseline_exit_slippage_price": exit_slip,
                    "baseline_price_pnl_jpy": gross,
                    "baseline_financing_jpy": finance,
                    "baseline_net_pnl_jpy": gross - finance,
                }
            )
        for position_id in sorted(set(current) - set(active)):
            position = current[position_id]
            active[position_id] = {
                **position,
                "open_frame_index": frame_index,
                "opened_phase": str(frame["phase"]),
                "first_seen_frame_index": frame_index,
            }
        for position_id in set(active) & set(current):
            if any(
                current[position_id][key] != active[position_id][key]
                for key in (
                    "pair",
                    "side",
                    "entry_price",
                    "tp_price",
                    "sl_price",
                    "opened_epoch",
                    "hard_exit_epoch",
                )
            ):
                raise DojoR13AIInventoryError(
                    "immutable position identity/geometry changed unexpectedly"
                )
            active[position_id]["units"] = current[position_id]["units"]
    if active:
        portfolio_result = _mapping(
            terminal.get("portfolio_result"),
            "terminal.portfolio_result",
        )
        if (
            terminal.get("terminal_policy") != "MONTH_END_FLAT_SETTLEMENT"
            or portfolio_result.get("terminal_policy")
            != "MONTH_END_FLAT_SETTLEMENT"
            or portfolio_result.get("open_position_count") != 0
        ):
            raise DojoR13AIInventoryError(
                "terminal transcript retained inventory without flat settlement"
            )
        frame = frames[-1]
        frame_index = len(frames) - 1
        for position_id in sorted(active):
            trade = active[position_id]
            fill_price, exit_slip = _close_fill(
                trade,
                frame,
                exit_slippage_price=slippage[trade["pair"]],
                protected_tp=False,
            )
            quotes = _quote_map(frame)
            gross = _trade_price_pnl_jpy(
                trade,
                float(trade["units"]),
                fill_price,
                quotes,
            )
            elapsed = max(
                0,
                int(frame["epoch"]) - int(trade["opened_epoch"]),
            )
            finance = (
                float(trade["units"])
                * financing[trade["pair"]][trade["side"]]
                * elapsed
                / 86400.0
            )
            completed.append(
                {
                    **trade,
                    "close_frame_index": frame_index,
                    "closed_epoch": int(frame["epoch"]),
                    "closed_phase": str(frame["phase"]),
                    "close_reason": "MONTH_END_FLAT",
                    "baseline_fill_price": fill_price,
                    "baseline_exit_slippage_price": exit_slip,
                    "baseline_price_pnl_jpy": gross,
                    "baseline_financing_jpy": finance,
                    "baseline_net_pnl_jpy": gross - finance,
                }
            )
    return completed


def prepare_r13_inputs(
    *,
    baseline_root: Path,
    job_id: str,
    output_root: Path,
    calibration_end_epoch: int,
) -> dict[str, Any]:
    """Extract a compact read-only derivative without rerunning bot economics."""

    baseline = baseline_root.resolve(strict=True)
    output = output_root.resolve()
    if output == baseline or baseline in output.parents:
        raise DojoR13AIInventoryError("output root must be outside immutable baseline")
    result_path = baseline / "job-results" / f"{job_id}.json"
    result = _read_json(result_path)
    completion = _read_json(baseline / "jobs" / job_id / "completion.json")
    if (
        result.get("job_status") != "COMPLETE"
        or result.get("complete_coordinate_count") != 12
        or result.get("failed_coordinate_count") != 0
        or completion.get("job_status") != "COMPLETE"
        or completion.get("month") != "2025-01"
        or completion.get("intrabar_path") != "OHLC"
        or completion.get("source_binding_id") != "M5_EXACT28_2020_2026H1"
    ):
        raise DojoR13AIInventoryError("r13 baseline identity/completion mismatch")
    job_result_size, job_result_file_sha = _file_sha256(result_path)
    runtime_path = baseline / "jobs" / job_id / "coordinate-runtimes.json"
    runtimes = _read_json(runtime_path)["coordinate_runtimes"]
    artifacts = {
        row["coordinate_id"]: row for row in result["economic_transcript_artifacts"]
    }
    coordinate_results = {
        row["coordinate_id"]: row for row in result["coordinate_results"]
    }
    if set(runtimes) != set(artifacts) or set(runtimes) != set(coordinate_results):
        raise DojoR13AIInventoryError("r13 coordinate denominator mismatch")

    frame_identity: list[tuple[int, str, int]] | None = None
    shared_frames: list[dict[str, Any]] | None = None
    coordinate_rows: list[dict[str, Any]] = []
    evidence_dir = baseline / "jobs" / job_id / "economic-evidence"
    for coordinate_id in sorted(runtimes):
        artifact = artifacts[coordinate_id]
        transcript_path = evidence_dir / artifact["transcript_filename"]
        header, frames, terminal, transcript_sha = _transcript_snapshot_rows(
            transcript_path,
            expected_file_sha256=artifact["transcript_file_sha256"],
            coordinate_id=coordinate_id,
        )
        identity = [
            (int(frame["epoch"]), str(frame["phase"]), int(frame["quote_watermark"]))
            for frame in frames
        ]
        if frame_identity is None:
            frame_identity = identity
            shared_frames = [
                {
                    "epoch": frame["epoch"],
                    "phase": frame["phase"],
                    "intrabar": frame["intrabar"],
                    "quote_watermark": frame["quote_watermark"],
                    "quotes": frame["quotes"],
                }
                for frame in frames
            ]
        elif identity != frame_identity:
            raise DojoR13AIInventoryError("coordinate quote clocks are not paired")
        cost_policy = header["portfolio_policy"]
        trades = _derive_trades(
            frames,
            cost_policy=cost_policy,
            terminal=terminal,
        )
        baseline_account_path = [
            {
                "epoch": int(frame["epoch"]),
                "phase": str(frame["phase"]),
                "balance_jpy": float(frame["account"]["balance_jpy"]),
                "equity_jpy": float(frame["account"]["equity_jpy"]),
                "margin_used_jpy": float(frame["account"]["margin_used_jpy"]),
            }
            for frame in frames
        ]
        terminal_result = _mapping(
            terminal["portfolio_result"],
            "terminal.portfolio_result",
        )
        baseline_account_path[-1] = {
            **baseline_account_path[-1],
            "balance_jpy": float(terminal_result["end_balance_jpy"]),
            "equity_jpy": float(terminal_result["end_equity_jpy"]),
            "margin_used_jpy": 0.0,
            "terminal_flat_settlement_applied": True,
        }
        runtime = runtimes[coordinate_id]
        family = runtime["portfolio_policy"]["active_worker_bindings"][0]["family_id"]
        coordinate_body = {
            "contract": PREPARED_CONTRACT,
            "schema_version": SCHEMA_VERSION,
            "coordinate_id": coordinate_id,
            "family_id": family,
            "cost_scenario": runtime["cost_scenario"],
            "transcript_filename": artifact["transcript_filename"],
            "transcript_file_sha256": transcript_sha,
            "transcript_file_bytes": artifact["transcript_file_bytes"],
            "policy_sha256": cost_policy["policy_sha256"],
            "cost_policy": cost_policy,
            "baseline_coordinate_result": coordinate_results[coordinate_id],
            "baseline_account_path": baseline_account_path,
            "trades": trades,
            "authority": dict(_AUTHORITY),
        }
        coordinate_body["prepared_coordinate_sha256"] = canonical_portfolio_sha256(
            coordinate_body
        )
        coordinate_file = output / "coordinates" / f"{coordinate_id}.json.gz"
        _atomic_gzip_json(coordinate_file, coordinate_body)
        file_bytes, file_sha = _file_sha256(coordinate_file)
        coordinate_rows.append(
            {
                "coordinate_id": coordinate_id,
                "family_id": family,
                "cost_scenario": runtime["cost_scenario"],
                "prepared_file": str(coordinate_file.relative_to(output)),
                "prepared_file_bytes": file_bytes,
                "prepared_file_sha256": file_sha,
                "trade_count": len(trades),
            }
        )
    if shared_frames is None or frame_identity is None:
        raise DojoR13AIInventoryError("no shared market frames were extracted")
    if not int(shared_frames[0]["epoch"]) < calibration_end_epoch < int(
        shared_frames[-1]["epoch"]
    ):
        raise DojoR13AIInventoryError("calibration boundary is outside January")
    frames_file = output / "market-frames.json.gz"
    _atomic_gzip_json(frames_file, shared_frames)
    frames_bytes, frames_sha = _file_sha256(frames_file)
    body = {
        "contract": STUDY_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "study_id": "r13-2025-01-ohlc-ai-inventory-oos-v1",
        "baseline_root": str(baseline),
        "baseline_job_id": job_id,
        "baseline_job_result_file_sha256": job_result_file_sha,
        "baseline_job_result_file_bytes": job_result_size,
        "month": "2025-01",
        "intrabar": "OHLC",
        "source_binding_id": "M5_EXACT28_2020_2026H1",
        "source_quote_coverage_proved": bool(
            result.get("source_quote_coverage_proved")
        ),
        "official_evidence_eligible": False,
        "paired_difference_classification": (
            "EXPERIMENTAL_SAME_INCOMPLETE_SOURCE_PAIRED_DIFFERENCE"
        ),
        "calibration_window": {
            "start_epoch": int(shared_frames[0]["epoch"]),
            "end_epoch": calibration_end_epoch,
        },
        "oos_window": {
            "start_epoch": calibration_end_epoch,
            "end_epoch": int(shared_frames[-1]["epoch"]) + 1,
        },
        "boundary_trade_policy": "PURGE_TRADES_OPENED_BEFORE_PARTITION_START",
        "initial_capital_jpy_per_partition": 200000.0,
        "market_frames_file": str(frames_file.relative_to(output)),
        "market_frames_file_bytes": frames_bytes,
        "market_frames_file_sha256": frames_sha,
        "coordinate_count": len(coordinate_rows),
        "coordinates": coordinate_rows,
        "immutable_baseline_was_reexecuted": False,
        "authority": dict(_AUTHORITY),
    }
    manifest = {**body, "study_sha256": canonical_portfolio_sha256(body)}
    _atomic_json(output / "study.json", manifest)
    return manifest


def load_prepared_study(output_root: Path) -> tuple[dict[str, Any], list[Any]]:
    root = output_root.resolve(strict=True)
    manifest = _read_json(root / "study.json")
    claimed = manifest.pop("study_sha256", None)
    if (
        manifest.get("contract") != STUDY_CONTRACT
        or claimed != canonical_portfolio_sha256(manifest)
    ):
        raise DojoR13AIInventoryError("prepared study manifest is invalid")
    manifest["study_sha256"] = claimed
    frames_path = root / manifest["market_frames_file"]
    size, digest = _file_sha256(frames_path)
    if (
        size != manifest["market_frames_file_bytes"]
        or digest != manifest["market_frames_file_sha256"]
    ):
        raise DojoR13AIInventoryError("prepared market frames changed")
    frames = _read_gzip_json(frames_path)
    return manifest, _sequence(frames, "market frames")


def load_prepared_coordinate(
    output_root: Path, manifest: Mapping[str, Any], coordinate_id: str
) -> dict[str, Any]:
    rows = {
        row["coordinate_id"]: row for row in manifest["coordinates"]
    }
    row = rows.get(coordinate_id)
    if row is None:
        raise DojoR13AIInventoryError("coordinate is outside prepared denominator")
    path = output_root.resolve(strict=True) / row["prepared_file"]
    size, digest = _file_sha256(path)
    if size != row["prepared_file_bytes"] or digest != row["prepared_file_sha256"]:
        raise DojoR13AIInventoryError("prepared coordinate changed")
    coordinate = _mapping(_read_gzip_json(path), "prepared coordinate")
    claimed = coordinate.pop("prepared_coordinate_sha256", None)
    if (
        coordinate.get("contract") != PREPARED_CONTRACT
        or claimed != canonical_portfolio_sha256(coordinate)
    ):
        raise DojoR13AIInventoryError("prepared coordinate digest mismatch")
    coordinate["prepared_coordinate_sha256"] = claimed
    return coordinate


def _returns(
    history: Mapping[str, Sequence[float]], pair: str, steps: int
) -> float | None:
    values = history.get(pair, ())
    if len(values) <= steps or values[-steps - 1] <= 0:
        return None
    return values[-1] / values[-steps - 1] - 1.0


def _technical_state(
    history: Mapping[str, Sequence[float]], pair: str
) -> dict[str, Any]:
    values = list(history.get(pair, ()))
    if not values:
        return {
            "return_5m": None,
            "return_15m": None,
            "return_30m": None,
            "return_60m": None,
            "return_120m": None,
            "realized_volatility_60m": None,
            "ema_fast_slow_gap": None,
            "range_position_60m": None,
        }
    returns = {
        minutes: _returns(history, pair, max(1, minutes // 5))
        for minutes in (5, 15, 30, 60, 120)
    }
    recent = values[-13:]
    log_returns = [
        math.log(later / earlier)
        for earlier, later in zip(recent, recent[1:])
        if earlier > 0 and later > 0
    ]
    volatility = (
        math.sqrt(sum(value * value for value in log_returns) / len(log_returns))
        if log_returns
        else None
    )
    fast = sum(values[-4:]) / min(4, len(values))
    slow = sum(values[-13:]) / min(13, len(values))
    range_values = values[-13:]
    low = min(range_values)
    high = max(range_values)
    position = (values[-1] - low) / (high - low) if high > low else 0.5
    return {
        "return_5m": returns[5],
        "return_15m": returns[15],
        "return_30m": returns[30],
        "return_60m": returns[60],
        "return_120m": returns[120],
        "realized_volatility_60m": volatility,
        "ema_fast_slow_gap": fast / slow - 1.0 if slow > 0 else None,
        "range_position_60m": position,
    }


def _inventory_packet(
    *,
    study_sha256: str,
    coordinate: Mapping[str, Any],
    arm: str,
    cadence_id: str,
    policy_version: str,
    prompt_version: str,
    frame: Mapping[str, Any],
    active_positions: Sequence[Mapping[str, Any]],
    realized_pnl_jpy: float,
    peak_equity_jpy: float,
    equity_jpy: float,
    history: Mapping[str, Sequence[float]],
    narrative_state: Mapping[str, Any] | None,
    triggers: Sequence[str],
    state_hash: str,
) -> dict[str, Any]:
    if arm not in {B_INVENTORY_ONLY, C_FORECAST_INVENTORY}:
        raise DojoR13AIInventoryError("AI packet arm is unsupported")
    quotes = _quote_map(frame)
    position_rows = []
    long_gross = 0.0
    short_gross = 0.0
    unrealized = 0.0
    oldest_age = 0
    pairs = []
    for position in active_positions:
        pair = position["pair"]
        pairs.append(pair)
        quote = quotes[pair]
        mark = float(quote["bid"] if position["side"] == "LONG" else quote["ask"])
        quote_pnl = float(position["remaining_units"]) * (
            mark - float(position["entry_price"])
            if position["side"] == "LONG"
            else float(position["entry_price"]) - mark
        )
        pnl = _quote_to_jpy(
            quote_pnl,
            pair.split("_", 1)[1],
            quotes,
        )
        unrealized += pnl
        notional = _quote_to_jpy(
            float(position["remaining_units"]) * mark,
            pair.split("_", 1)[1],
            quotes,
        )
        if position["side"] == "LONG":
            long_gross += notional
        else:
            short_gross += notional
        age = max(0, int(frame["epoch"]) - int(position["opened_epoch"]))
        oldest_age = max(oldest_age, age)
        risk_distance = abs(
            float(position["entry_price"]) - float(position["sl_price"])
        )
        reward_distance = (
            abs(float(position["tp_price"]) - float(position["entry_price"]))
            if position["tp_price"] is not None
            else None
        )
        favorable = (
            mark - float(position["entry_price"])
            if position["side"] == "LONG"
            else float(position["entry_price"]) - mark
        )
        tp_progress = (
            favorable / reward_distance
            if reward_distance is not None and reward_distance > 0
            else None
        )
        loss_progress = max(0.0, -favorable / risk_distance) if risk_distance else 0.0
        position_rows.append(
            {
                "position_id": position["position_id"],
                "pair": pair,
                "side": position["side"],
                "remaining_units": position["remaining_units"],
                "entry_price": position["entry_price"],
                "mark_price": mark,
                "tp_price": position["tp_price"],
                "sl_price": position["sl_price"],
                "unrealized_pnl_jpy": pnl,
                "age_seconds": age,
                "tp_progress": tp_progress,
                "loss_progress": loss_progress,
                "mfe_jpy": position["mfe_jpy"],
                "giveback_jpy": max(0.0, position["mfe_jpy"] - pnl),
                "technical": _technical_state(history, pair),
            }
        )
    primary_pair = pairs[0] if pairs else coordinate["family_id"]
    packet_body = {
        "contract": PACKET_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "study_sha256": study_sha256,
        "coordinate_id": coordinate["coordinate_id"],
        "family_id": coordinate["family_id"],
        "cost_scenario": coordinate["cost_scenario"],
        "arm": arm,
        "cadence_id": cadence_id,
        "policy_version": policy_version,
        "policy_guardrails": _copy_json(
            PHASE_A_POLICY_PROFILES.get(
                policy_version,
                PHASE_A_POLICY_PROFILES["BALANCED_V1"],
            )
        ),
        "prompt_version": prompt_version,
        "observed_at_epoch": int(frame["epoch"]),
        "observed_at": _utc_iso(int(frame["epoch"])),
        "cutoff_epoch": int(frame["epoch"]),
        "market_clock_only": True,
        "append_wall_clock_included": False,
        "future_quote_included": False,
        "terminal_result_included": False,
        "other_arm_result_included": False,
        "trigger_ids": sorted(set(triggers)),
        "state_hash": state_hash,
        "inventory": {
            "positions": position_rows,
            "long_gross_exposure_jpy": long_gross,
            "short_gross_exposure_jpy": short_gross,
            "gross_exposure_jpy": long_gross + short_gross,
            "net_exposure_jpy": long_gross - short_gross,
            "margin_utilization_fraction": (
                (long_gross + short_gross) / 25.0 / max(equity_jpy, 1.0)
            ),
            "unrealized_pnl_jpy": unrealized,
            "realized_pnl_jpy": realized_pnl_jpy,
            "drawdown_fraction": max(
                0.0, (peak_equity_jpy - equity_jpy) / max(peak_equity_jpy, 1.0)
            ),
            "oldest_position_age_seconds": oldest_age,
            "position_concentration_fraction": 1.0 if position_rows else 0.0,
        },
        "observed_market": {
            "primary_pair": primary_pair,
            "quote_age_seconds_by_position_pair": {
                pair: int(
                    frame.get("quote_age_seconds_by_pair", {}).get(pair, 0)
                )
                for pair in sorted(set(pairs))
            },
            "technical_by_position_pair": {
                pair: _technical_state(history, pair) for pair in sorted(set(pairs))
            },
        },
        "prior_narrative_state": (
            _copy_json(narrative_state) if narrative_state is not None else None
        ),
        "allowed_actions": list(ACTIONS),
        "allowed_direction_restrictions": list(DIRECTION_RESTRICTIONS),
        "authority": dict(_AUTHORITY),
    }
    return {
        **packet_body,
        "packet_sha256": canonical_portfolio_sha256(packet_body),
    }


def validate_worker_response(
    *,
    packet: Mapping[str, Any],
    response: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed on schema, authority, arm, cutoff, or evidence mismatch."""

    packet_row = _mapping(packet, "packet")
    packet_sha = _sha256(packet_row.get("packet_sha256"), "packet.packet_sha256")
    packet_body = {
        key: value for key, value in packet_row.items() if key != "packet_sha256"
    }
    if (
        packet_row.get("contract") != PACKET_CONTRACT
        or packet_sha != canonical_portfolio_sha256(packet_body)
        or packet_row.get("future_quote_included") is not False
        or packet_row.get("terminal_result_included") is not False
        or packet_row.get("other_arm_result_included") is not False
        or packet_row.get("append_wall_clock_included") is not False
    ):
        raise DojoR13AIInventoryError("packet is not a causal sealed packet")
    row = _mapping(response, "response")
    _exact(
        row,
        {
            "contract",
            "schema_version",
            "packet_sha256",
            "observed_at",
            "narrative_state",
            "forecast",
            "inventory_diagnosis",
            "action",
            "rationale",
            "next_trigger",
            "authority",
        },
        "response",
    )
    if (
        row["contract"] != RESPONSE_CONTRACT
        or row["schema_version"] != SCHEMA_VERSION
        or row["packet_sha256"] != packet_sha
        or row["observed_at"] != packet_row["observed_at"]
    ):
        raise DojoR13AIInventoryError("response is bound to the wrong packet/cutoff")
    authority = _mapping(row["authority"], "response.authority")
    if authority != _AUTHORITY:
        raise DojoR13AIInventoryError("worker attempted to change authority")
    narrative = _mapping(row["narrative_state"], "narrative_state")
    _exact(
        narrative,
        {
            "version",
            "micro_concrete",
            "micro_abstract",
            "macro_concrete",
            "macro_abstract",
            "global_story",
            "strategy_story",
            "prior_hypothesis",
            "current_observation",
            "what_matched",
            "what_failed",
            "why",
            "next_hypothesis",
            "evidence_refs",
        },
        "narrative_state",
    )
    _integer(narrative["version"], "narrative_state.version", minimum=1)
    for field_name in (
        "micro_concrete",
        "micro_abstract",
        "macro_concrete",
        "macro_abstract",
        "global_story",
        "strategy_story",
        "prior_hypothesis",
        "current_observation",
        "what_matched",
        "what_failed",
        "why",
        "next_hypothesis",
    ):
        _identifier(narrative[field_name], f"narrative_state.{field_name}", maximum=800)
    refs = [
        _identifier(value, "narrative evidence ref", maximum=160)
        for value in _sequence(narrative["evidence_refs"], "narrative evidence refs")
    ]
    if not refs:
        raise DojoR13AIInventoryError("narrative requires observed evidence refs")
    forecast = row["forecast"]
    if packet_row["arm"] == B_INVENTORY_ONLY:
        if forecast is not None:
            raise DojoR13AIInventoryError("inventory-only arm must not emit forecast")
    else:
        forecast = _mapping(forecast, "forecast")
        _exact(
            forecast,
            {
                "direction",
                "confidence",
                "horizon_min",
                "invalidation",
                "evidence_refs",
            },
            "forecast",
        )
        if forecast["direction"] not in FORECAST_DIRECTIONS:
            raise DojoR13AIInventoryError("forecast direction is unsupported")
        _finite(forecast["confidence"], "forecast.confidence", minimum=0, maximum=1)
        if forecast["horizon_min"] not in {30, 60, 120}:
            raise DojoR13AIInventoryError("forecast horizon is unsupported")
        _identifier(forecast["invalidation"], "forecast.invalidation", maximum=500)
        if not _sequence(forecast["evidence_refs"], "forecast.evidence_refs"):
            raise DojoR13AIInventoryError("forecast requires observed evidence refs")
    diagnosis = _mapping(row["inventory_diagnosis"], "inventory_diagnosis")
    _exact(
        diagnosis,
        {
            "risk_level",
            "strategy_regime_fit",
            "inventory_story_mismatch",
            "tp_profit_retention_risk",
            "loss_giveback_risk",
        },
        "inventory_diagnosis",
    )
    if diagnosis["risk_level"] not in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}:
        raise DojoR13AIInventoryError("inventory risk level is unsupported")
    if diagnosis["strategy_regime_fit"] not in {"FIT", "MIXED", "MISMATCH", "UNKNOWN"}:
        raise DojoR13AIInventoryError("strategy/regime fit is unsupported")
    for field_name in (
        "inventory_story_mismatch",
        "tp_profit_retention_risk",
        "loss_giveback_risk",
    ):
        _identifier(diagnosis[field_name], f"inventory_diagnosis.{field_name}", maximum=500)
    action = _mapping(row["action"], "action")
    _exact(
        action,
        {"type", "fraction", "direction_restriction"},
        "action",
    )
    if action["type"] not in ACTIONS:
        raise DojoR13AIInventoryError("worker action is unsupported")
    if action["direction_restriction"] not in DIRECTION_RESTRICTIONS:
        raise DojoR13AIInventoryError("direction restriction is unsupported")
    if action["type"] in {
        "REDUCE_LONG",
        "REDUCE_SHORT",
        "PARTIAL_CLOSE",
    }:
        _finite(action["fraction"], "action.fraction", minimum=0.1, maximum=0.9)
    elif action["fraction"] is not None:
        raise DojoR13AIInventoryError("fraction is only valid for reductions")
    _identifier(row["rationale"], "rationale", maximum=1200)
    _identifier(row["next_trigger"], "next_trigger", maximum=500)
    sealed = {
        **row,
        "narrative_state": {**narrative, "evidence_refs": refs},
        "response_sha256": canonical_portfolio_sha256(row),
    }
    return sealed


@dataclass
class _OverlayPosition:
    trade: dict[str, Any]
    remaining_units: float
    realized_pnl_jpy: float = 0.0
    financing_jpy: float = 0.0
    mfe_jpy: float = 0.0
    mae_jpy: float = 0.0
    closed: bool = False
    skipped: bool = False
    intervention_count: int = 0

    def packet_row(self) -> dict[str, Any]:
        return {
            **self.trade,
            "remaining_units": self.remaining_units,
            "mfe_jpy": self.mfe_jpy,
            "mae_jpy": self.mae_jpy,
        }


@dataclass
class _SimulationState:
    cash_jpy: float
    peak_equity_jpy: float
    minimum_equity_jpy: float
    max_drawdown_fraction: float = 0.0
    max_margin_utilization_fraction: float = 0.0
    margin_call_count: int = 0
    margin_call_active: bool = False
    realized_pnl_jpy: float = 0.0
    financing_jpy: float = 0.0
    active: dict[str, _OverlayPosition] = field(default_factory=dict)
    pause_new_entries: bool = False
    direction_restriction: str = "NONE"
    last_decision_epoch: int | None = None
    last_state_hash: str | None = None
    narrative_state: dict[str, Any] | None = None
    decision_count: int = 0
    actual_ai_call_count: int = 0
    fallback_count: int = 0
    call_cap_exhausted: bool = False
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0
    interventions: list[dict[str, Any]] = field(default_factory=list)
    forecasts: list[dict[str, Any]] = field(default_factory=list)
    executed_trade_ids: set[str] = field(default_factory=set)
    skipped_trade_ids: set[str] = field(default_factory=set)
    closed_trade_results: list[dict[str, Any]] = field(default_factory=list)
    turnover_units: float = 0.0
    turnover_jpy: float = 0.0
    immutable_background_cashflow_jpy: float = 0.0


def _mark_position(
    position: _OverlayPosition, frame: Mapping[str, Any]
) -> float:
    quotes = _quote_map(frame)
    trade = position.trade
    quote = quotes[trade["pair"]]
    mark = float(quote["bid"] if trade["side"] == "LONG" else quote["ask"])
    return _trade_price_pnl_jpy(
        trade, position.remaining_units, mark, quotes
    )


def _equity(state: _SimulationState, frame: Mapping[str, Any]) -> float:
    return state.cash_jpy + sum(
        _mark_position(position, frame) for position in state.active.values()
    )


def _mark_extrema(state: _SimulationState, frame: Mapping[str, Any]) -> None:
    for position in state.active.values():
        mark = _mark_position(position, frame)
        position.mfe_jpy = max(position.mfe_jpy, mark)
        position.mae_jpy = min(position.mae_jpy, mark)


def _margin_utilization(
    state: _SimulationState,
    frame: Mapping[str, Any],
    *,
    leverage: float,
    equity_jpy: float,
) -> float:
    quotes = _quote_map(frame)
    gross_notional = 0.0
    for position in state.active.values():
        trade = position.trade
        quote = quotes[trade["pair"]]
        mark = float(
            quote["bid"] if trade["side"] == "LONG" else quote["ask"]
        )
        gross_notional += _quote_to_jpy(
            position.remaining_units * mark,
            trade["pair"].split("_", 1)[1],
            quotes,
        )
    return gross_notional / max(leverage, 1.0) / max(equity_jpy, 1.0)


def _apply_financing(
    state: _SimulationState,
    *,
    elapsed_seconds: int,
    financing_by_pair: Mapping[str, Mapping[str, float]],
) -> None:
    if elapsed_seconds <= 0:
        return
    for position in state.active.values():
        trade = position.trade
        cost = (
            position.remaining_units
            * financing_by_pair[trade["pair"]][trade["side"]]
            * elapsed_seconds
            / 86400.0
        )
        position.financing_jpy += cost
        state.financing_jpy += cost
        state.cash_jpy -= cost


def _baseline_financing_rate_by_epoch(
    *,
    trades: Sequence[Mapping[str, Any]],
    financing_by_pair: Mapping[str, Mapping[str, float]],
    epochs: Sequence[int],
) -> dict[int, float]:
    opens = sorted(trades, key=lambda trade: int(trade["opened_epoch"]))
    closes = sorted(trades, key=lambda trade: int(trade["closed_epoch"]))
    active_rates: dict[str, float] = {}
    open_index = 0
    close_index = 0
    total_rate = 0.0
    result: dict[int, float] = {}
    for epoch in sorted(set(epochs)):
        while (
            open_index < len(opens)
            and int(opens[open_index]["opened_epoch"]) < epoch
        ):
            trade = opens[open_index]
            rate = (
                float(trade["units"])
                * financing_by_pair[trade["pair"]][trade["side"]]
            )
            active_rates[trade["position_id"]] = rate
            total_rate += rate
            open_index += 1
        while (
            close_index < len(closes)
            and int(closes[close_index]["closed_epoch"]) < epoch
        ):
            trade = closes[close_index]
            total_rate -= active_rates.pop(trade["position_id"], 0.0)
            close_index += 1
        result[epoch] = total_rate
    return result


def _close_overlay_units(
    state: _SimulationState,
    position: _OverlayPosition,
    *,
    frame: Mapping[str, Any],
    units: float,
    reason: str,
    exit_slippage_price: float,
    protected_tp: bool = False,
) -> float:
    units = min(max(0.0, units), position.remaining_units)
    if units <= 0:
        return 0.0
    fill, _ = _close_fill(
        position.trade,
        frame,
        exit_slippage_price=exit_slippage_price,
        protected_tp=protected_tp,
    )
    pnl = _trade_price_pnl_jpy(
        position.trade, units, fill, _quote_map(frame)
    )
    position.remaining_units -= units
    position.realized_pnl_jpy += pnl
    position.intervention_count += int(reason.startswith("AI_"))
    state.cash_jpy += pnl
    state.realized_pnl_jpy += pnl
    if position.remaining_units <= 1e-9:
        position.closed = True
        state.active.pop(position.trade["position_id"], None)
        state.closed_trade_results.append(
            {
                "position_id": position.trade["position_id"],
                "pair": position.trade["pair"],
                "side": position.trade["side"],
                "reason": reason,
                "net_pnl_jpy": (
                    position.realized_pnl_jpy - position.financing_jpy
                ),
                "baseline_net_pnl_jpy": position.trade[
                    "baseline_net_pnl_jpy"
                ],
                "mfe_jpy": position.mfe_jpy,
                "mae_jpy": position.mae_jpy,
                "intervention_count": position.intervention_count,
            }
        )
    return pnl


def _event_triggers(
    state: _SimulationState,
    *,
    frame: Mapping[str, Any],
    equity_jpy: float,
) -> tuple[list[str], str, bool]:
    triggers: set[str] = set()
    high_risk = False
    risk_rows = []
    for position in state.active.values():
        mark = _mark_position(position, frame)
        trade = position.trade
        pair_quote = _quote_map(frame)[trade["pair"]]
        price = float(
            pair_quote["bid"] if trade["side"] == "LONG" else pair_quote["ask"]
        )
        favorable = (
            price - float(trade["entry_price"])
            if trade["side"] == "LONG"
            else float(trade["entry_price"]) - price
        )
        risk_distance = abs(
            float(trade["entry_price"]) - float(trade["sl_price"])
        )
        reward_distance = (
            abs(float(trade["tp_price"]) - float(trade["entry_price"]))
            if trade["tp_price"] is not None
            else None
        )
        loss_progress = max(0.0, -favorable / risk_distance) if risk_distance else 0
        tp_progress = (
            max(0.0, favorable / reward_distance)
            if reward_distance is not None and reward_distance > 0
            else 0.0
        )
        giveback_fraction = (
            max(0.0, position.mfe_jpy - mark) / position.mfe_jpy
            if position.mfe_jpy > 0
            else 0.0
        )
        age = max(0, int(frame["epoch"]) - int(trade["opened_epoch"]))
        loss_bucket = min(3, int(loss_progress / 0.25))
        tp_bucket = min(3, int(tp_progress / 0.25))
        giveback_bucket = min(3, int(giveback_fraction / 0.2))
        age_bucket = min(3, age // 900)
        if loss_bucket >= 1:
            triggers.add("LOSS_PROGRESS")
        if tp_bucket >= 1:
            triggers.add("TP_PROGRESS")
        if giveback_bucket >= 1:
            triggers.add("PROFIT_GIVEBACK")
        if age_bucket >= 2:
            triggers.add("POSITION_AGE")
        if loss_bucket >= 2 or giveback_bucket >= 2 or age_bucket >= 3:
            high_risk = True
        risk_rows.append(
            {
                "position_id": trade["position_id"],
                "loss_bucket": loss_bucket,
                "tp_bucket": tp_bucket,
                "giveback_bucket": giveback_bucket,
                "age_bucket": age_bucket,
            }
        )
    drawdown = max(0.0, (state.peak_equity_jpy - equity_jpy) / state.peak_equity_jpy)
    drawdown_bucket = min(4, int(drawdown / 0.025))
    if drawdown_bucket >= 1:
        triggers.add("DRAWDOWN")
    if drawdown_bucket >= 2:
        high_risk = True
    state_body = {
        "positions": risk_rows,
        "pause": state.pause_new_entries,
        "direction_restriction": state.direction_restriction,
        "drawdown_bucket": drawdown_bucket,
    }
    state_hash = canonical_portfolio_sha256(state_body)
    if state.last_state_hash is None and state.active:
        triggers.add("NEW_INVENTORY")
    elif state.last_state_hash != state_hash:
        triggers.add("STATE_CHANGE")
    return sorted(triggers), state_hash, high_risk


def _cadence_due(
    *,
    cadence_id: str,
    state: _SimulationState,
    epoch: int,
    triggers: Sequence[str],
    high_risk: bool,
) -> bool:
    if not state.active and not state.pause_new_entries:
        return False
    elapsed = (
        None if state.last_decision_epoch is None else epoch - state.last_decision_epoch
    )
    if cadence_id in CADENCE_SECONDS and CADENCE_SECONDS[cadence_id] is not None:
        interval = int(CADENCE_SECONDS[cadence_id] or 0)
        return elapsed is None or elapsed >= interval
    material_events = {
        "NEW_INVENTORY",
        "LOSS_PROGRESS",
        "TP_PROGRESS",
        "PROFIT_GIVEBACK",
        "DRAWDOWN",
        "POSITION_AGE",
    }
    event_due = bool(set(triggers) & material_events)
    if cadence_id == "EVENT_DRIVEN":
        return event_due
    if cadence_id == "ADAPTIVE":
        interval = 900 if high_risk else 3600
        return event_due or elapsed is None or elapsed >= interval
    raise DojoR13AIInventoryError("cadence is unsupported")


def _default_narrative(version: int, evidence_refs: Sequence[str]) -> dict[str, Any]:
    return {
        "version": version,
        "micro_concrete": "Observed inventory and completed-price indicators only.",
        "micro_abstract": "Short-horizon risk is mixed.",
        "macro_concrete": "No external macro feed is present in this replay packet.",
        "macro_abstract": "Macro state remains unknown.",
        "global_story": "Preserve capital when inventory evidence deteriorates.",
        "strategy_story": "Evaluate strategy-regime fit from observed tape.",
        "prior_hypothesis": "No prior hypothesis." if version == 1 else "Prior state carried.",
        "current_observation": "Current causal packet reviewed.",
        "what_matched": "Inventory state was available.",
        "what_failed": "No confirmed mismatch.",
        "why": "Only observed evidence is admissible.",
        "next_hypothesis": "Reassess on the next registered trigger.",
        "evidence_refs": list(evidence_refs),
    }


def deterministic_worker_response(
    packet: Mapping[str, Any],
    *,
    policy_id: str,
) -> dict[str, Any]:
    """Deterministic Phase-A filter used only for calibration/cadence selection."""

    profile = PHASE_A_POLICY_PROFILES.get(
        policy_id,
        PHASE_A_POLICY_PROFILES["BALANCED_V1"],
    )
    positions = packet["inventory"]["positions"]
    worst_loss = max((row["loss_progress"] for row in positions), default=0.0)
    worst_giveback = max(
        (
            row["giveback_jpy"] / row["mfe_jpy"]
            if row["mfe_jpy"] > 0
            else 0.0
            for row in positions
        ),
        default=0.0,
    )
    max_tp = max(
        (
            row["tp_progress"]
            for row in positions
            if row["tp_progress"] is not None
        ),
        default=0.0,
    )
    action_type = "HOLD"
    fraction: float | None = None
    risk_level = "LOW"
    if worst_loss >= profile["close_loss_progress"]:
        action_type = "CLOSE_RISKY"
        risk_level = "CRITICAL"
    elif (
        max_tp >= profile["tp_progress_trigger"]
        and worst_giveback >= profile["giveback_trigger"]
    ):
        action_type = "PARTIAL_CLOSE"
        fraction = 0.5
        risk_level = "HIGH"
    elif worst_loss >= profile["partial_loss_progress"]:
        action_type = "PARTIAL_CLOSE"
        fraction = 0.5
        risk_level = "HIGH"
    elif (
        worst_loss >= profile["partial_loss_progress"] / 2.0
        or worst_giveback >= profile["giveback_trigger"] / 2.0
    ):
        risk_level = "MEDIUM"

    forecast: dict[str, Any] | None = None
    direction_restriction = "NONE"
    if packet["arm"] == C_FORECAST_INVENTORY:
        technical_rows = packet["observed_market"]["technical_by_position_pair"]
        signals = []
        for technical in technical_rows.values():
            for key in ("return_30m", "return_60m", "ema_fast_slow_gap"):
                value = technical.get(key)
                if value is not None:
                    signals.append(float(value))
        aggregate = sum(signals)
        scale = sum(abs(value) for value in signals)
        if not signals or scale < 1e-9:
            direction = "UNCERTAIN"
            confidence = 0.35
        elif abs(aggregate) < 0.2 * scale:
            direction = "RANGE"
            confidence = 0.55
        elif aggregate > 0:
            direction = "UP"
            confidence = min(0.82, 0.5 + abs(aggregate) / max(scale, 1e-9) * 0.3)
            direction_restriction = "LONG_ONLY"
        else:
            direction = "DOWN"
            confidence = min(0.82, 0.5 + abs(aggregate) / max(scale, 1e-9) * 0.3)
            direction_restriction = "SHORT_ONLY"
        forecast = {
            "direction": direction,
            "confidence": confidence,
            "horizon_min": 60,
            "invalidation": "Reassess when 30m return changes sign or risk bucket changes.",
            "evidence_refs": ["observed_market.technical_by_position_pair"],
        }
        mismatch = any(
            (row["side"] == "LONG" and direction == "DOWN")
            or (row["side"] == "SHORT" and direction == "UP")
            for row in positions
        )
        if mismatch and confidence >= 0.65:
            action_type = "CLOSE_RISKY" if worst_loss >= 0.35 else "PARTIAL_CLOSE"
            fraction = None if action_type == "CLOSE_RISKY" else 0.5
            risk_level = "HIGH"
    version = (
        int(packet["prior_narrative_state"]["version"]) + 1
        if packet["prior_narrative_state"] is not None
        else 1
    )
    response = {
        "contract": RESPONSE_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "packet_sha256": packet["packet_sha256"],
        "observed_at": packet["observed_at"],
        "narrative_state": _default_narrative(
            version,
            ["inventory", "observed_market.technical_by_position_pair"],
        ),
        "forecast": forecast,
        "inventory_diagnosis": {
            "risk_level": risk_level,
            "strategy_regime_fit": "MIXED",
            "inventory_story_mismatch": "No high-confidence mismatch." if risk_level == "LOW" else "Risk evidence deteriorated.",
            "tp_profit_retention_risk": f"max_tp_progress={max_tp:.4f}",
            "loss_giveback_risk": f"loss_progress={worst_loss:.4f};giveback={worst_giveback:.4f}",
        },
        "action": {
            "type": action_type,
            "fraction": fraction,
            "direction_restriction": direction_restriction,
        },
        "rationale": f"Phase-A deterministic policy {policy_id}.",
        "next_trigger": "Next registered cadence or material state-change trigger.",
        "authority": dict(_AUTHORITY),
    }
    # Exercise the same fail-closed validator during Phase A, but return the
    # unsealed worker payload because the orchestrator owns response sealing.
    validate_worker_response(packet=packet, response=response)
    return response


def preregistered_hold_response(
    packet: Mapping[str, Any],
    *,
    reason: str,
    direction_restriction: str = "NONE",
) -> dict[str, Any]:
    response = deterministic_worker_response(
        packet,
        policy_id="PREREGISTERED_FAIL_CLOSED_HOLD",
    )
    response["action"] = {
        "type": "HOLD",
        "fraction": None,
        "direction_restriction": direction_restriction,
    }
    if packet["arm"] == C_FORECAST_INVENTORY:
        response["forecast"] = {
            "direction": "UNCERTAIN",
            "confidence": 0.0,
            "horizon_min": 60,
            "invalidation": "Reassess at the next registered trigger.",
            "evidence_refs": ["inventory"],
        }
    response["rationale"] = reason
    validate_worker_response(packet=packet, response=response)
    return response


WorkerCallback = Callable[[dict[str, Any]], Mapping[str, Any]]


def score_forecasts_posthoc(
    *,
    forecast_rows: Sequence[Mapping[str, Any]],
    frames: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Score sealed forecasts after acting has finished.

    Target quotes are consulted only here, never by the packet builder, worker,
    cadence trigger, or action path.
    """

    close_frames = [
        frame for frame in frames if str(frame["phase"]) == "C"
    ]
    by_epoch = {int(frame["epoch"]): frame for frame in close_frames}
    scored: list[dict[str, Any]] = []
    for forecast in forecast_rows:
        pair = forecast.get("pair")
        if not pair:
            continue
        observed_epoch = int(forecast["observed_at_epoch"])
        horizon_min = int(forecast["horizon_min"])
        target_epoch = observed_epoch + horizon_min * 60
        observed_frame = by_epoch.get(observed_epoch)
        target_frame = by_epoch.get(target_epoch)
        if observed_frame is None or target_frame is None:
            continue
        observed_quote = _quote_map(observed_frame).get(str(pair))
        target_quote = _quote_map(target_frame).get(str(pair))
        if observed_quote is None or target_quote is None:
            continue
        observed_mid = (
            float(observed_quote["bid"]) + float(observed_quote["ask"])
        ) / 2.0
        target_mid = (
            float(target_quote["bid"]) + float(target_quote["ask"])
        ) / 2.0
        realized_return = target_mid / observed_mid - 1.0
        if realized_return > 0.0001:
            actual = "UP"
        elif realized_return < -0.0001:
            actual = "DOWN"
        else:
            actual = "RANGE"
        direction = str(forecast["direction"])
        confidence = float(forecast["confidence"])
        hit = direction == actual
        outcomes = ("UP", "DOWN", "RANGE")
        if direction in outcomes:
            probabilities = {
                outcome: (
                    confidence
                    if outcome == direction
                    else (1.0 - confidence) / 2.0
                )
                for outcome in outcomes
            }
        else:
            probabilities = {outcome: 1.0 / 3.0 for outcome in outcomes}
        brier = sum(
            (probabilities[outcome] - float(outcome == actual)) ** 2
            for outcome in outcomes
        )
        actual_probability = max(probabilities[actual], 1e-12)
        scored.append(
            {
                "observed_at_epoch": observed_epoch,
                "target_epoch": target_epoch,
                "pair": pair,
                "horizon_min": horizon_min,
                "direction": direction,
                "confidence": confidence,
                "actual_direction": actual,
                "realized_return": realized_return,
                "direction_hit": hit,
                "brier_score": brier,
                "log_loss": -math.log(actual_probability),
                "fallback": bool(forecast.get("fallback", False)),
            }
        )
    by_horizon: dict[str, Any] = {}
    for horizon in (30, 60, 120):
        rows = [row for row in scored if row["horizon_min"] == horizon]
        by_horizon[str(horizon)] = {
            "scored_count": len(rows),
            "direction_accuracy": (
                sum(row["direction_hit"] for row in rows) / len(rows)
                if rows
                else None
            ),
            "mean_brier_score": (
                sum(row["brier_score"] for row in rows) / len(rows)
                if rows
                else None
            ),
            "mean_log_loss": (
                sum(row["log_loss"] for row in rows) / len(rows)
                if rows
                else None
            ),
        }
    calibration_bins = []
    for lower in (0.0, 0.25, 0.5, 0.75):
        upper = lower + 0.25
        rows = [
            row
            for row in scored
            if lower <= row["confidence"] < upper
            or (upper == 1.0 and row["confidence"] == 1.0)
        ]
        calibration_bins.append(
            {
                "lower": lower,
                "upper": upper,
                "count": len(rows),
                "mean_confidence": (
                    sum(row["confidence"] for row in rows) / len(rows)
                    if rows
                    else None
                ),
                "empirical_accuracy": (
                    sum(row["direction_hit"] for row in rows) / len(rows)
                    if rows
                    else None
                ),
            }
        )
    high_confidence = [row for row in scored if row["confidence"] >= 0.7]
    return {
        "scored_count": len(scored),
        "unscored_count": len(forecast_rows) - len(scored),
        "direction_accuracy": (
            sum(row["direction_hit"] for row in scored) / len(scored)
            if scored
            else None
        ),
        "confidence_calibration_mae": (
            sum(
                abs(row["confidence"] - float(row["direction_hit"]))
                for row in scored
            )
            / len(scored)
            if scored
            else None
        ),
        "mean_brier_score": (
            sum(row["brier_score"] for row in scored) / len(scored)
            if scored
            else None
        ),
        "mean_log_loss": (
            sum(row["log_loss"] for row in scored) / len(scored)
            if scored
            else None
        ),
        "high_confidence_forecast_count": len(high_confidence),
        "wrong_high_confidence_forecast_rate": (
            sum(not row["direction_hit"] for row in high_confidence)
            / len(high_confidence)
            if high_confidence
            else None
        ),
        "by_horizon": by_horizon,
        "calibration_bins": calibration_bins,
        "scored_rows": scored,
        "posthoc_only": True,
        "acting_policy_input": False,
    }


def simulate_partition(
    *,
    study: Mapping[str, Any],
    coordinate: Mapping[str, Any],
    frames: Sequence[Mapping[str, Any]],
    partition: str,
    arm: str,
    cadence_id: str | None,
    policy_version: str,
    prompt_version: str,
    worker: WorkerCallback | None,
    max_ai_calls: int | None = None,
    capture_full_audit: bool = True,
) -> dict[str, Any]:
    """Simulate one paired overlay cell over calibration or held-out OOS."""

    if partition not in {"CALIBRATION", "OOS"}:
        raise DojoR13AIInventoryError("partition is unsupported")
    if arm not in ARMS:
        raise DojoR13AIInventoryError("arm is unsupported")
    if arm == A_BOT_ONLY:
        if cadence_id is not None or worker is not None:
            raise DojoR13AIInventoryError("bot-only cell cannot call a worker")
    elif cadence_id not in CADENCE_SECONDS or worker is None:
        raise DojoR13AIInventoryError("AI cell requires cadence and worker")
    window = (
        study["calibration_window"]
        if partition == "CALIBRATION"
        else study["oos_window"]
    )
    start_epoch = int(window["start_epoch"])
    end_epoch = int(window["end_epoch"])
    partition_frames = [
        frame for frame in frames if start_epoch <= int(frame["epoch"]) < end_epoch
    ]
    if not partition_frames:
        raise DojoR13AIInventoryError("partition has no market frames")
    trades = [
        _copy_json(trade)
        for trade in coordinate["trades"]
        if start_epoch <= int(trade["opened_epoch"]) < end_epoch
        and int(trade["closed_epoch"]) < end_epoch
    ]
    purged_boundary = sum(
        int(trade["opened_epoch"]) < start_epoch <= int(trade["closed_epoch"])
        for trade in coordinate["trades"]
    )
    entries: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    exits: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    all_baseline_exits: dict[
        tuple[int, str], list[dict[str, Any]]
    ] = defaultdict(list)
    for trade in trades:
        entries[
            (int(trade["opened_epoch"]), str(trade["opened_phase"]))
        ].append(trade)
        exits[(int(trade["closed_epoch"]), str(trade["closed_phase"]))].append(trade)
    for trade in coordinate["trades"]:
        all_baseline_exits[
            (int(trade["closed_epoch"]), str(trade["closed_phase"]))
        ].append(trade)

    initial_capital = float(study["initial_capital_jpy_per_partition"])
    state = _SimulationState(
        cash_jpy=initial_capital,
        peak_equity_jpy=initial_capital,
        minimum_equity_jpy=initial_capital,
    )
    financing_rows = {
        row["pair"]: {
            "LONG": float(row["long_cost_jpy_per_unit_day"]),
            "SHORT": float(row["short_cost_jpy_per_unit_day"]),
        }
        for row in coordinate["cost_policy"]["financing_by_pair"]
    }
    baseline_account_by_coordinate = {
        (int(row["epoch"]), str(row["phase"])): row
        for row in coordinate["baseline_account_path"]
    }
    baseline_financing_rate = _baseline_financing_rate_by_epoch(
        trades=coordinate["trades"],
        financing_by_pair=financing_rows,
        epochs=[int(frame["epoch"]) for frame in partition_frames],
    )
    exit_slippage = {
        row["pair"]: float(row["exit_slippage_price"])
        for row in coordinate["cost_policy"]["slippage_by_pair"]
    }
    history: dict[str, list[float]] = defaultdict(list)
    quote_cache: dict[str, dict[str, float]] = {}
    quote_observed_epoch: dict[str, int] = {}
    previous_epoch: int | None = None
    previous_baseline_balance: float | None = None
    equity_path: list[float] = []

    for frame in partition_frames:
        epoch = int(frame["epoch"])
        phase = str(frame["phase"])
        fresh_quotes = _quote_map(frame)
        for pair, quote in fresh_quotes.items():
            quote_cache[pair] = quote
            quote_observed_epoch[pair] = epoch
        frame = {
            **frame,
            "quotes": [
                {"pair": pair, **quote}
                for pair, quote in sorted(quote_cache.items())
            ],
            "fresh_quote_pairs": sorted(fresh_quotes),
            "quote_age_seconds_by_pair": {
                pair: max(0, epoch - quote_observed_epoch[pair])
                for pair in sorted(quote_cache)
            },
        }
        baseline_account = baseline_account_by_coordinate.get((epoch, phase))
        if baseline_account is None:
            raise DojoR13AIInventoryError(
                "prepared baseline account path is incomplete"
            )
        baseline_balance = float(baseline_account["balance_jpy"])
        if previous_baseline_balance is not None:
            baseline_balance_delta = (
                baseline_balance - previous_baseline_balance
            )
            baseline_close_pnl = sum(
                float(trade["baseline_price_pnl_jpy"])
                for trade in all_baseline_exits.get((epoch, phase), ())
            )
            elapsed_for_baseline = (
                max(0, epoch - previous_epoch)
                if previous_epoch is not None
                else 0
            )
            baseline_financing = (
                baseline_financing_rate[epoch]
                * elapsed_for_baseline
                / 86400.0
            )
            background_cashflow = (
                baseline_balance_delta
                - baseline_close_pnl
                + baseline_financing
            )
            state.cash_jpy += background_cashflow
            state.immutable_background_cashflow_jpy += background_cashflow
        previous_baseline_balance = baseline_balance
        if previous_epoch is not None:
            _apply_financing(
                state,
                elapsed_seconds=max(0, epoch - previous_epoch),
                financing_by_pair=financing_rows,
            )
        previous_epoch = epoch
        frame_key = (epoch, phase)

        for trade in exits.get(frame_key, ()):
            position = state.active.get(trade["position_id"])
            if position is None:
                continue
            _close_overlay_units(
                state,
                position,
                frame=frame,
                units=position.remaining_units,
                reason=f"BASELINE_{trade['close_reason']}",
                exit_slippage_price=exit_slippage[trade["pair"]],
                protected_tp=trade["close_reason"] == "TAKE_PROFIT",
            )

        if phase == "C":
            for pair, quote in quote_cache.items():
                history[pair].append((quote["bid"] + quote["ask"]) / 2.0)

        # The immutable schedule proposes entries at the close coordinate.  A
        # previously established PAUSE/direction restriction can veto them;
        # accepted inventory is visible to the same-coordinate event trigger.
        for trade in entries.get(frame_key, ()):
            allowed_side = (
                state.direction_restriction == "NONE"
                or (
                    state.direction_restriction == "LONG_ONLY"
                    and trade["side"] == "LONG"
                )
                or (
                    state.direction_restriction == "SHORT_ONLY"
                    and trade["side"] == "SHORT"
                )
                or (
                    state.direction_restriction == "NO_NEW_LONGS"
                    and trade["side"] != "LONG"
                )
                or (
                    state.direction_restriction == "NO_NEW_SHORTS"
                    and trade["side"] != "SHORT"
                )
            )
            if state.pause_new_entries or not allowed_side:
                state.skipped_trade_ids.add(trade["position_id"])
                continue
            state.executed_trade_ids.add(trade["position_id"])
            state.turnover_units += float(trade["units"])
            state.turnover_jpy += _quote_to_jpy(
                float(trade["units"]) * float(trade["entry_price"]),
                str(trade["pair"]).split("_", 1)[1],
                _quote_map(frame),
            )
            state.active[trade["position_id"]] = _OverlayPosition(
                trade=trade,
                remaining_units=float(trade["units"]),
            )

        _mark_extrema(state, frame)
        equity_before = _equity(state, frame)
        state.peak_equity_jpy = max(state.peak_equity_jpy, equity_before)
        state.minimum_equity_jpy = min(state.minimum_equity_jpy, equity_before)
        state.max_drawdown_fraction = max(
            state.max_drawdown_fraction,
            (state.peak_equity_jpy - equity_before)
            / max(state.peak_equity_jpy, 1.0),
        )
        margin_utilization = _margin_utilization(
            state,
            frame,
            leverage=float(coordinate["cost_policy"]["leverage"]),
            equity_jpy=equity_before,
        )
        state.max_margin_utilization_fraction = max(
            state.max_margin_utilization_fraction,
            margin_utilization,
        )
        closeout_fraction = float(
            coordinate["cost_policy"]["margin_closeout_fraction"]
        )
        if margin_utilization >= closeout_fraction and not state.margin_call_active:
            state.margin_call_count += 1
            state.margin_call_active = True
        elif margin_utilization < closeout_fraction:
            state.margin_call_active = False
        if arm != A_BOT_ONLY and phase == "C":
            triggers, state_hash, high_risk = _event_triggers(
                state, frame=frame, equity_jpy=equity_before
            )
            if _cadence_due(
                cadence_id=str(cadence_id),
                state=state,
                epoch=epoch,
                triggers=triggers,
                high_risk=high_risk,
            ):
                elapsed_since_decision = (
                    None
                    if state.last_decision_epoch is None
                    else epoch - state.last_decision_epoch
                )
                adaptive_cache_hold = (
                    cadence_id == "ADAPTIVE"
                    and state_hash == state.last_state_hash
                    and elapsed_since_decision is not None
                    and elapsed_since_decision < (900 if high_risk else 3600)
                )
                low_risk_cache_hold = (
                    state_hash == state.last_state_hash and not high_risk
                )
                if (
                    state_hash == state.last_state_hash
                    and cadence_id == "EVENT_DRIVEN"
                ) or adaptive_cache_hold or low_risk_cache_hold or (
                    state.call_cap_exhausted
                ):
                    pass
                else:
                    packet = _inventory_packet(
                        study_sha256=study["study_sha256"],
                        coordinate=coordinate,
                        arm=arm,
                        cadence_id=str(cadence_id),
                        policy_version=policy_version,
                        prompt_version=prompt_version,
                        frame=frame,
                        active_positions=[
                            item.packet_row() for item in state.active.values()
                        ],
                        realized_pnl_jpy=state.realized_pnl_jpy,
                        peak_equity_jpy=state.peak_equity_jpy,
                        equity_jpy=equity_before,
                        history=history,
                        narrative_state=state.narrative_state,
                        triggers=triggers,
                        state_hash=state_hash,
                    )
                    fallback = False
                    failure_class: str | None = None
                    attempted_worker_response_sha256: str | None = None
                    if (
                        max_ai_calls is not None
                        and state.actual_ai_call_count >= max_ai_calls
                    ):
                        fallback = True
                        failure_class = "PREREGISTERED_CALL_CAP"
                        state.call_cap_exhausted = True
                        raw_response = preregistered_hold_response(
                            packet,
                            reason=(
                                "AI call cap reached; preregistered HOLD fallback."
                            ),
                            direction_restriction=state.direction_restriction,
                        )
                        response = validate_worker_response(
                            packet=packet,
                            response=raw_response,
                        )
                    else:
                        state.actual_ai_call_count += 1
                        try:
                            raw_response = worker(packet)
                            if isinstance(raw_response, Mapping):
                                attempted_worker_response_sha256 = (
                                    canonical_portfolio_sha256(
                                        _mapping(
                                            raw_response,
                                            "attempted worker response",
                                        )
                                    )
                                )
                            response = validate_worker_response(
                                packet=packet,
                                response=raw_response,
                            )
                        except (
                            DojoR13AIInventoryError,
                            RuntimeError,
                            TimeoutError,
                            TypeError,
                            ValueError,
                        ) as exc:
                            fallback = True
                            failure_class = type(exc).__name__
                            raw_response = preregistered_hold_response(
                                packet,
                                reason=(
                                    "Worker failure; preregistered HOLD fallback."
                                ),
                                direction_restriction=state.direction_restriction,
                            )
                            response = validate_worker_response(
                                packet=packet,
                                response=raw_response,
                            )
                    state.fallback_count += int(fallback)
                    state.decision_count += 1
                    state.last_decision_epoch = epoch
                    state.last_state_hash = state_hash
                    state.narrative_state = response["narrative_state"]
                    packet_bytes = len(
                        json.dumps(packet, ensure_ascii=False, separators=(",", ":"))
                    )
                    response_bytes = len(
                        json.dumps(response, ensure_ascii=False, separators=(",", ":"))
                    )
                    state.estimated_input_tokens += math.ceil(packet_bytes / 4)
                    state.estimated_output_tokens += math.ceil(response_bytes / 4)
                    action = response["action"]
                    state.direction_restriction = action["direction_restriction"]
                    if action["type"] == "PAUSE_NEW_ENTRIES":
                        state.pause_new_entries = True
                    elif action["type"] == "RESUME":
                        state.pause_new_entries = False
                    targets = list(state.active.values())
                    if action["type"] == "CLOSE_RISKY":
                        targets = [
                            position
                            for position in targets
                            if _mark_position(position, frame) < 0
                        ]
                    elif action["type"] == "REDUCE_LONG":
                        targets = [
                            position
                            for position in targets
                            if position.trade["side"] == "LONG"
                        ]
                    elif action["type"] == "REDUCE_SHORT":
                        targets = [
                            position
                            for position in targets
                            if position.trade["side"] == "SHORT"
                        ]
                    if action["type"] in {
                        "CLOSE_RISKY",
                        "CLOSE_ALL",
                        "REDUCE_LONG",
                        "REDUCE_SHORT",
                        "PARTIAL_CLOSE",
                    }:
                        for position in targets:
                            fraction = (
                                1.0
                                if action["type"] in {"CLOSE_RISKY", "CLOSE_ALL"}
                                else float(action["fraction"])
                            )
                            _close_overlay_units(
                                state,
                                position,
                                frame=frame,
                                units=position.remaining_units * fraction,
                                reason=f"AI_{action['type']}",
                                exit_slippage_price=exit_slippage[
                                    position.trade["pair"]
                                ],
                            )
                    audit = {
                        "packet_sha256": packet["packet_sha256"],
                        "cutoff_epoch": packet["cutoff_epoch"],
                        "prompt_version": prompt_version,
                        "policy_version": policy_version,
                        "response_sha256": response["response_sha256"],
                        "attempted_worker_response_sha256": (
                            attempted_worker_response_sha256
                        ),
                        "trigger_ids": triggers,
                        "action": action,
                        "fallback": fallback,
                        "failure_class": failure_class,
                    }
                    if capture_full_audit:
                        state.interventions.append(audit)
                    if capture_full_audit and response["forecast"] is not None:
                        forecast_pair = (
                            packet["inventory"]["positions"][0]["pair"]
                            if packet["inventory"]["positions"]
                            else None
                        )
                        state.forecasts.append(
                            {
                                "observed_at_epoch": epoch,
                                "pair": forecast_pair,
                                **response["forecast"],
                                "packet_sha256": packet["packet_sha256"],
                                "fallback": fallback,
                            }
                        )

        equity_after = _equity(state, frame)
        state.peak_equity_jpy = max(state.peak_equity_jpy, equity_after)
        state.minimum_equity_jpy = min(state.minimum_equity_jpy, equity_after)
        state.max_drawdown_fraction = max(
            state.max_drawdown_fraction,
            (state.peak_equity_jpy - equity_after)
            / max(state.peak_equity_jpy, 1.0),
        )
        equity_path.append(equity_after)

    terminal_frame = partition_frames[-1]
    for position in list(state.active.values()):
        _close_overlay_units(
            state,
            position,
            frame=terminal_frame,
            units=position.remaining_units,
            reason="PARTITION_END_FLAT",
            exit_slippage_price=exit_slippage[position.trade["pair"]],
        )
    end_equity = state.cash_jpy
    trade_results = state.closed_trade_results
    wins = [row for row in trade_results if row["net_pnl_jpy"] > 0]
    losses = [row for row in trade_results if row["net_pnl_jpy"] < 0]
    gross_profit = sum(row["net_pnl_jpy"] for row in wins)
    gross_loss = -sum(row["net_pnl_jpy"] for row in losses)
    baseline_by_id = {
        trade["position_id"]: float(trade["baseline_net_pnl_jpy"])
        for trade in trades
    }
    candidate_by_id = {
        row["position_id"]: float(row["net_pnl_jpy"]) for row in trade_results
    }
    loss_avoided = sum(
        max(0.0, candidate_by_id.get(position_id, 0.0) - baseline_net)
        for position_id, baseline_net in baseline_by_id.items()
        if baseline_net < 0
    )
    missed_upside = sum(
        max(0.0, baseline_net - candidate_by_id.get(position_id, 0.0))
        for position_id, baseline_net in baseline_by_id.items()
        if baseline_net > 0
    )
    retained_values = [
        max(0.0, row["net_pnl_jpy"]) / row["mfe_jpy"]
        for row in trade_results
        if row["mfe_jpy"] > 0
    ]
    forecast_evaluation = score_forecasts_posthoc(
        forecast_rows=state.forecasts,
        frames=partition_frames,
    )
    actual_worker_forecasts = [
        row for row in state.forecasts if not row.get("fallback", False)
    ]
    actual_worker_forecast_evaluation = score_forecasts_posthoc(
        forecast_rows=actual_worker_forecasts,
        frames=partition_frames,
    )
    estimated_cost_usd = (
        state.estimated_input_tokens * 5.0 / 1_000_000.0
        + state.estimated_output_tokens * 15.0 / 1_000_000.0
    )
    metrics = {
        "net_after_all_costs_jpy": end_equity - initial_capital,
        "ending_equity_jpy": end_equity,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else None,
        "win_rate": len(wins) / len(trade_results) if trade_results else 0.0,
        "expectancy_jpy": (
            (end_equity - initial_capital) / len(trades)
            if trades
            else 0.0
        ),
        "expectancy_per_executed_trade_jpy": (
            (end_equity - initial_capital) / len(trade_results)
            if trade_results
            else 0.0
        ),
        "max_drawdown_fraction": state.max_drawdown_fraction,
        "minimum_equity_jpy": state.minimum_equity_jpy,
        "max_margin_utilization_fraction": (
            state.max_margin_utilization_fraction
        ),
        "margin_call_count": state.margin_call_count,
        "ruin_event_count": int(state.minimum_equity_jpy <= 0),
        "scheduled_trade_count": len(trades),
        "trade_count": len(trade_results),
        "skipped_trade_count": len(state.skipped_trade_ids),
        "tp_profit_retained_fraction": (
            sum(retained_values) / len(retained_values) if retained_values else 0.0
        ),
        "loss_avoided_jpy": loss_avoided,
        "missed_upside_jpy": missed_upside,
        "turnover_units": state.turnover_units,
        "turnover_jpy": state.turnover_jpy,
        "immutable_unobservable_background_cashflow_jpy": (
            state.immutable_background_cashflow_jpy
        ),
        "ai_decision_count": state.decision_count,
        "ai_call_count": state.actual_ai_call_count,
        "ai_fallback_count": state.fallback_count,
        "ai_estimated_input_tokens": state.estimated_input_tokens,
        "ai_estimated_output_tokens": state.estimated_output_tokens,
        "ai_notional_cost_usd": estimated_cost_usd,
        "ai_cost_assumption": (
            "NOTIONAL_ONLY_USD_5_PER_M_INPUT_15_PER_M_OUTPUT"
        ),
    }
    body = {
        "contract": CELL_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "study_sha256": study["study_sha256"],
        "prepared_coordinate_sha256": coordinate["prepared_coordinate_sha256"],
        "coordinate_id": coordinate["coordinate_id"],
        "family_id": coordinate["family_id"],
        "cost_scenario": coordinate["cost_scenario"],
        "partition": partition,
        "arm": arm,
        "cadence_id": cadence_id,
        "policy_version": policy_version,
        "prompt_version": prompt_version,
        "status": "COMPLETE",
        "partition_start_epoch": start_epoch,
        "partition_end_epoch": end_epoch,
        "initial_capital_jpy": initial_capital,
        "source_quote_coverage_proved": False,
        "purged_boundary_trade_count": purged_boundary,
        "metrics": metrics,
        "intervention_audit": state.interventions,
        "forecast_rows": state.forecasts,
        "forecast_evaluation_all": forecast_evaluation,
        "forecast_evaluation_actual_worker": (
            actual_worker_forecast_evaluation
        ),
        "authority": dict(_AUTHORITY),
    }
    return {**body, "cell_sha256": canonical_portfolio_sha256(body)}


__all__ = [
    "A_BOT_ONLY",
    "ACTIONS",
    "ARMS",
    "B_INVENTORY_ONLY",
    "CADENCE_SECONDS",
    "C_FORECAST_INVENTORY",
    "CELL_CONTRACT",
    "DojoR13AIInventoryError",
    "PACKET_CONTRACT",
    "RESPONSE_CONTRACT",
    "STUDY_CONTRACT",
    "deterministic_worker_response",
    "load_prepared_coordinate",
    "load_prepared_study",
    "prepare_r13_inputs",
    "score_forecasts_posthoc",
    "simulate_partition",
    "validate_worker_response",
]
