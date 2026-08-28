#!/usr/bin/env python3
"""Offline, review-gated replay for M5_EMA_STATE_IMPULSE_INVENTORY_V1.

No network, credential, broker, order, launchd, or Git interface exists here.
The default audit opens no candle file.  The one-shot result path is locked
until an independent receipt binds this runner and the immutable preregistration.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[2]
PREREG_PATH = ROOT / "PREREGISTRATION.json"
REVIEW_PATH = ROOT / "INDEPENDENT_REVIEW.json"
RESULT_PATH = ROOT / "result.json"
PACKET_PATH = ROOT / "evidence_packet.json"
SIGNALS_PATH = ROOT / "signals.jsonl"
TRADES_PATH = ROOT / "trades.jsonl"
OUTPUTS = (RESULT_PATH, PACKET_PATH, SIGNALS_PATH, TRADES_PATH)
ARMS = ("RAW_SIGNAL", "EXECUTABLE_BASE", "ADVERSE_STRESS")
UTC = dt.timezone.utc
BAR_SECONDS = 300
WINDOW_BARS = 7
EXPECTED_PREREG_FILE_SHA256 = "0311d644dcd33cfb642181ab0d4965f74923ee031122a3b856ec187403d3bb35"
EXPECTED_PREREG_CANONICAL_SHA256 = "e0867db99d04e64a01e4ced62b812c4ff09bc7cfe35a7cfbe459fc636819148d"


def canonical(value) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                   allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: Iterable[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical(row).decode("utf-8") + "\n")
    temporary.replace(path)


def parse_time(value: str) -> int:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("timestamp must be UTC-Z text")
    body = value[:-1]
    if "." in body:
        head, fraction = body.split(".", 1)
        body = head + "." + fraction[:6]
    parsed = dt.datetime.fromisoformat(body + "+00:00")
    if parsed.tzinfo is None:
        raise ValueError("timestamp must be timezone aware")
    return int(parsed.timestamp())


def iso_utc(seconds: int) -> str:
    return dt.datetime.fromtimestamp(seconds, UTC).strftime(
        "%Y-%m-%dT%H:%M:%S.000000Z"
    )


def pip_size(pair: str) -> float:
    return 0.01 if pair.endswith("_JPY") else 0.0001


def signed_pips(pair: str, side: str, entry: float, exit_: float) -> float:
    direction = 1.0 if side == "LONG" else -1.0
    return direction * (exit_ - entry) / pip_size(pair)


@dataclass(frozen=True, slots=True)
class Bar:
    time: int
    bid: tuple[float, float, float, float]
    ask: tuple[float, float, float, float]
    volume: int
    source_hash: str

    def side(self, name: str, field: str) -> float:
        index = {"o": 0, "h": 1, "l": 2, "c": 3}[field]
        return (self.bid if name == "bid" else self.ask)[index]

    def mid(self, field: str) -> float:
        return (self.side("bid", field) + self.side("ask", field)) / 2.0


@dataclass(frozen=True, slots=True)
class Signal:
    signal_id: str
    pair: str
    side: str
    decision_time: int
    decision_bar_time: int
    expected_fill_time: int
    decision_bar_hash: str
    fast_ema: float
    slow_ema: float
    slow_slope: float
    momentum_price: float
    atr_price: float
    observed_spread_price: float
    tp_distance_price: float
    direction_correct_at_six_bars: bool | None


@dataclass(slots=True)
class Pending:
    signal: Signal
    due_time: int


@dataclass(slots=True)
class Position:
    signal: Signal
    arm: str
    entry_time: int
    entry_price: float
    entry_mid: float
    tp_price: float
    max_age_due_time: int
    entry_spread_pips: float
    entry_slippage_pips: float
    latency_proxy_pips: float
    mfe_pips: float = 0.0
    mae_pips: float = 0.0
    bars_observed: int = 0


@dataclass(slots=True)
class ArmState:
    arm: str
    pending: dict[str, Pending] = field(default_factory=dict)
    positions: dict[str, Position] = field(default_factory=dict)
    trades: list[dict] = field(default_factory=list)
    outcomes: dict[str, dict] = field(default_factory=dict)
    realized_jpy: float = 0.0
    missing_jpy_conversions: int = 0
    inventory_samples: list[int] = field(default_factory=list)
    equity_marks: list[tuple[int, float]] = field(default_factory=list)


def load_preregistration() -> dict:
    payload = PREREG_PATH.read_bytes()
    if sha_bytes(payload) != EXPECTED_PREREG_FILE_SHA256:
        raise ValueError("preregistration file SHA-256 mismatch")
    value = json.loads(payload)
    if sha_bytes(canonical(value)) != EXPECTED_PREREG_CANONICAL_SHA256:
        raise ValueError("preregistration canonical SHA-256 mismatch")
    return value


def validate_preregistration(prereg: dict) -> dict[str, bool]:
    strategy = prereg["strategy"]
    checks = {
        "candidate": prereg["candidate_id"]
        == "M5_EMA_STATE_IMPULSE_INVENTORY_V1_HISTORICAL_REPLAY_V1",
        "runtime_strategy": prereg["runtime_strategy_id"]
        == "M5_EMA_STATE_IMPULSE_INVENTORY_V1",
        "status": prereg["status"]
        == "PREREGISTERED_AWAITING_INDEPENDENT_REVIEW",
        "exact_prior_absent": prereg["prior_evidence_search"]["exact_hits"] == 0
        and prereg["prior_evidence_search"]["exact_historical_evidence_found"] is False,
        "fixed_config": (
            strategy["fast_ema_bars"], strategy["slow_ema_bars"],
            strategy["momentum_bars"], strategy["atr_bars"],
            strategy["tp_atr_multiple"], strategy["tp_spread_multiple_floor"],
            strategy["max_age_bars"], strategy["virtual_units"],
        ) == (3, 6, 3, 6, 0.5, 1.5, 6, 1000),
        "capacity": strategy["hard_max_open_positions_per_instrument"] == 1
        and strategy["hard_max_open_positions_total"] == 2,
        "no_cost_gate": strategy["entry_cost_gate_used"] is False
        and prereg["execution_arms"]["cost_is_entry_gate"] is False,
        "arms": all(arm in prereg["execution_arms"] for arm in ARMS),
        "strict_chronology": prereg["chronology"]["same_timestamp_open_is_eligible"] is False
        and prereg["chronology"]["strict_fill_lag_bars_from_decision_bar"] == 2,
        "boundary": prereg["input"]["semantic_decoder_exclusive_end_utc"]
        == "2025-08-28T04:05:00.000000Z"
        and prereg["input"]["rows_or_bytes_after_boundary_allowed"] == 0,
        "single_hypothesis": prereg["statistics"]["hypothesis_family_size"] == 1,
        "bootstrap": prereg["statistics"]["bootstrap_resamples"] == 10000
        and prereg["statistics"]["bootstrap_seed"] == 20260828,
        "review_required": prereg["review_and_execution_gate"][
            "result_execution_before_independent_review"
        ] is False,
        "authority": prereg["authority"] == {
            "offline_only": True,
            "network_attempts_allowed": 0,
            "credential_reads_allowed": 0,
            "broker_or_account_access_allowed": False,
            "external_order_attempts_allowed": 0,
            "external_orders_allowed": 0,
            "launchd_actions_allowed": 0,
            "git_actions_allowed": 0,
            "running_shadow_files_mutable": False,
        },
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("preregistration mismatch: " + ",".join(failed))
    return checks


def audit_local_contracts(prereg: dict) -> dict:
    """Verify small local contracts without opening any candle file."""
    dataset_root = REPO_ROOT / prereg["input"]["dataset_root"]
    paths = {
        "runtime_contract": REPO_ROOT / prereg["runtime_binding"]["contract_path"],
        "paper_execution": REPO_ROOT / prereg["runtime_binding"]["paper_execution_path"],
        "manifest": dataset_root / "manifest.json",
        "gap_report": dataset_root / "gap_report.json",
    }
    expected = {
        "runtime_contract": prereg["runtime_binding"]["contract_file_sha256"],
        "paper_execution": prereg["runtime_binding"]["paper_execution_file_sha256"],
        "manifest": prereg["input"]["manifest_sha256"],
        "gap_report": prereg["input"]["gap_report_sha256"],
    }
    observed = {name: sha_file(path) for name, path in paths.items()}
    if observed != expected:
        raise ValueError("local immutable contract hash mismatch")
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    if manifest.get("canonical_dataset_sha256") != prereg["input"][
        "canonical_dataset_sha256"
    ]:
        raise ValueError("canonical dataset seal mismatch")
    if manifest.get("external_order_attempts") != 0 or manifest.get(
        "external_orders"
    ) != 0:
        raise ValueError("capture authority mismatch")
    runtime_contract = json.loads(paths["runtime_contract"].read_text(encoding="utf-8"))
    strategy_sha = sha_bytes(canonical(runtime_contract["paper_execution"]))
    if strategy_sha != prereg["runtime_binding"][
        "paper_execution_subobject_canonical_sha256"
    ]:
        raise ValueError("runtime paper_execution subobject mismatch")
    return {
        "contract_hashes": observed,
        "runtime_paper_execution_canonical_sha256": strategy_sha,
        "candle_files_opened": 0,
        "rows_decoded": 0,
        "post_boundary_bytes_read": 0,
        "network_attempts": 0,
        "credential_reads": 0,
        "external_order_attempts": 0,
        "external_orders": 0,
    }


def validate_review_receipt(prereg: dict, runner_path: Path | None = None) -> dict:
    if not REVIEW_PATH.exists():
        raise PermissionError("INDEPENDENT_REVIEW_REQUIRED_BEFORE_ANY_CANDLE_DECODE")
    receipt = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))
    runner = runner_path or Path(__file__).resolve()
    expected = {
        "status": "APPROVED_FOR_OFFLINE_REPLAY",
        "preregistration_file_sha256": sha_file(PREREG_PATH),
        "preregistration_canonical_sha256": sha_bytes(canonical(prereg)),
        "runner_file_sha256": sha_file(runner),
        "p0_findings": 0,
        "p1_findings": 0,
        "reviewer_is_implementation_lane": False,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise PermissionError("independent review receipt mismatch: " + key)
    if not isinstance(receipt.get("reviewer_task_id"), str) or not receipt[
        "reviewer_task_id"
    ]:
        raise PermissionError("independent reviewer identity missing")
    return receipt


def read_exact_prefix(path: Path, length: int, expected_sha: str) -> bytes:
    """Read exactly the sealed prefix; never stat or read one suffix byte."""
    if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
        raise ValueError("invalid prefix length")
    remaining = length
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    with Path(path).open("rb", buffering=0) as handle:
        while remaining:
            chunk = handle.read(min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError("source ended before preregistered byte boundary")
            digest.update(chunk)
            chunks.append(chunk)
            remaining -= len(chunk)
    if digest.hexdigest() != expected_sha:
        raise ValueError("sealed prefix SHA-256 mismatch")
    return b"".join(chunks)


def read_exact_suffix(path: Path, start: dict, end: dict) -> bytes:
    prefix = read_exact_prefix(path, start["exclusive_byte_offset"], start["prefix_sha256"])
    del prefix
    payload = read_exact_prefix(path, end["exclusive_byte_offset"], end["prefix_sha256"])
    return payload[start["exclusive_byte_offset"]:]


def parse_bar(raw: bytes, pair: str, minimum: int, maximum: int) -> Bar:
    row = json.loads(raw)
    identity = (
        row.get("schema"), row.get("instrument"), row.get("granularity"),
        row.get("price_component"), row.get("complete"),
    )
    if identity != ("QR_OANDA_HISTORICAL_M5_BA_ROW_V1", pair, "M5", "BA", True):
        raise ValueError("invalid completed M5 BID/ASK candle identity")
    timestamp = parse_time(row.get("time_utc"))
    if not minimum <= timestamp < maximum:
        raise AssertionError("semantic decode outside authorized phase")
    if row.get("volume_semantics") != "OANDA_PRICE_COUNT_NOT_TRADED_VOLUME":
        raise ValueError("volume semantics changed")
    volume = row.get("volume")
    if isinstance(volume, bool) or not isinstance(volume, int) or volume < 0:
        raise ValueError("invalid price-count volume")
    decoded = []
    for side_name in ("bid", "ask"):
        side = row.get(side_name)
        if not isinstance(side, dict) or set(side) != {"o", "h", "l", "c"}:
            raise ValueError("invalid BID/ASK OHLC shape")
        values = tuple(float(side[key]) for key in ("o", "h", "l", "c"))
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError("nonfinite or nonpositive price")
        if not values[2] <= min(values[0], values[3]) <= max(
            values[0], values[3]
        ) <= values[1]:
            raise ValueError("invalid OHLC geometry")
        decoded.append(values)
    bid, ask = decoded
    if any(bid[index] > ask[index] for index in range(4)):
        raise ValueError("crossed BID/ASK candle")
    return Bar(timestamp, bid, ask, volume, sha_bytes(canonical(row)))


def _validate_increasing(rows: list[Bar], pair: str) -> None:
    if any(rows[index].time >= rows[index + 1].time for index in range(len(rows) - 1)):
        raise ValueError("duplicate or out-of-order rows: " + pair)


def load_discovery_bytes(prereg: dict, audit: dict) -> dict[str, list[Bar]]:
    root = REPO_ROOT / prereg["input"]["dataset_root"]
    start = parse_time(prereg["splits"]["calibration"]["from_utc"])
    end = parse_time(prereg["splits"]["discovery"]["to_utc"])
    output = {}
    for pair in prereg["input"]["symbols"]:
        contract = prereg["input"]["discovery_prefix_contract"][pair]
        payload = read_exact_prefix(root / contract["path"], contract[
            "exclusive_byte_offset"
        ], contract["prefix_sha256"])
        lines = payload.splitlines()
        if len(lines) != contract["prefix_rows"]:
            raise ValueError("discovery prefix row mismatch: " + pair)
        rows = [parse_bar(line, pair, start, end) for line in lines]
        _validate_increasing(rows, pair)
        output[pair] = rows
        audit["pairs"][pair] = {
            "discovery_prefix_bytes_read": len(payload),
            "discovery_rows_decoded": len(rows),
            "validation_rows_decoded": 0,
        }
    audit["candle_files_opened"] = len(output)
    audit["candle_prefix_open_operations"] = len(output)
    audit["rows_decoded"] = sum(len(rows) for rows in output.values())
    return output


def load_validation_bytes(prereg: dict, audit: dict, discovery_lock: str) -> dict[str, list[Bar]]:
    if not discovery_lock or any(
        entry["validation_rows_decoded"] for entry in audit["pairs"].values()
    ):
        raise AssertionError("fixed discovery lock required before validation decode")
    root = REPO_ROOT / prereg["input"]["dataset_root"]
    start = parse_time(prereg["splits"]["locked_validation"]["from_utc"])
    end = parse_time(prereg["splits"]["locked_validation"]["to_utc"])
    output = {}
    for pair in prereg["input"]["symbols"]:
        first = prereg["input"]["discovery_prefix_contract"][pair]
        final = prereg["input"]["validation_prefix_contract"][pair]
        payload = read_exact_suffix(root / final["path"], first, final)
        lines = payload.splitlines()
        expected = final["prefix_rows"] - first["prefix_rows"]
        if len(lines) != expected:
            raise ValueError("validation suffix row mismatch: " + pair)
        rows = [parse_bar(line, pair, start, end) for line in lines]
        _validate_increasing(rows, pair)
        output[pair] = rows
        audit["pairs"][pair]["validation_suffix_bytes_read"] = len(payload)
        audit["pairs"][pair]["validation_rows_decoded"] = len(rows)
    audit["candle_prefix_open_operations"] += 2 * len(output)
    audit["rows_decoded"] += sum(len(rows) for rows in output.values())
    audit["discovery_lock_before_validation_decode"] = True
    audit["discovery_lock_sha256"] = discovery_lock
    return output


def _ema_rows(windows: np.ndarray, period: int) -> np.ndarray:
    alpha = 2.0 / (period + 1.0)
    result = windows[:, 0].astype(np.float64, copy=True)
    for column in range(1, windows.shape[1]):
        result = alpha * windows[:, column] + (1.0 - alpha) * result
    return result


def build_signals(
    pair: str,
    rows: list[Bar],
    phase_start: int,
    phase_end: int,
    prereg_canonical_sha: str,
) -> list[Signal]:
    """Vectorize the exact seven-bar runtime feature pass."""
    if len(rows) < WINDOW_BARS:
        return []
    times = np.asarray([row.time for row in rows], dtype=np.int64)
    bid = np.asarray([row.bid for row in rows], dtype=np.float64)
    ask = np.asarray([row.ask for row in rows], dtype=np.float64)
    mid = (bid + ask) / 2.0
    time_windows = np.lib.stride_tricks.sliding_window_view(times, WINDOW_BARS)
    close_windows = np.lib.stride_tricks.sliding_window_view(mid[:, 3], WINDOW_BARS)
    high_windows = np.lib.stride_tricks.sliding_window_view(mid[:, 1], WINDOW_BARS)
    low_windows = np.lib.stride_tricks.sliding_window_view(mid[:, 2], WINDOW_BARS)
    contiguous = np.all(np.diff(time_windows, axis=1) == BAR_SECONDS, axis=1)
    fast = _ema_rows(close_windows, 3)
    slow = _ema_rows(close_windows, 6)
    prior_slow = _ema_rows(close_windows[:, :-1], 6)
    momentum = close_windows[:, -1] - close_windows[:, -4]
    slope = slow - prior_slow
    direction = np.sign(fast - slow).astype(np.int8)
    true_range = np.maximum.reduce((
        high_windows[:, 1:] - low_windows[:, 1:],
        np.abs(high_windows[:, 1:] - close_windows[:, :-1]),
        np.abs(low_windows[:, 1:] - close_windows[:, :-1]),
    ))
    atr = np.mean(true_range, axis=1)
    end_indices = np.arange(WINDOW_BARS - 1, len(rows), dtype=np.int64)
    spread = ask[end_indices, 3] - bid[end_indices, 3]
    decision = times[end_indices] + BAR_SECONDS
    eligible = (
        contiguous
        & (direction != 0)
        & (direction * momentum > 0)
        & (direction * slope > 0)
        & np.isfinite(atr) & (atr > 0)
        & np.isfinite(spread) & (spread > 0)
        & (decision >= phase_start) & (decision < phase_end)
    )
    time_index = {row.time: index for index, row in enumerate(rows)}
    signals = []
    for vector_index in np.flatnonzero(eligible):
        row_index = int(end_indices[vector_index])
        decision_time = int(decision[vector_index])
        expected_fill = decision_time + BAR_SECONDS
        direction_value = int(direction[vector_index])
        side = "LONG" if direction_value > 0 else "SHORT"
        horizon_index = time_index.get(expected_fill + 5 * BAR_SECONDS)
        direction_correct = None
        if horizon_index is not None:
            entry_index = time_index.get(expected_fill)
            if entry_index is not None:
                direction_correct = signed_pips(
                    pair, side, rows[entry_index].mid("o"), rows[horizon_index].mid("c")
                ) > 0
        material = "|".join((
            prereg_canonical_sha,
            "M5_EMA_STATE_IMPULSE_INVENTORY_V1",
            pair,
            iso_utc(decision_time),
            rows[row_index].source_hash,
            side,
        )).encode("utf-8")
        signals.append(Signal(
            signal_id=sha_bytes(material), pair=pair, side=side,
            decision_time=decision_time,
            decision_bar_time=rows[row_index].time,
            expected_fill_time=expected_fill,
            decision_bar_hash=rows[row_index].source_hash,
            fast_ema=float(fast[vector_index]), slow_ema=float(slow[vector_index]),
            slow_slope=float(slope[vector_index]),
            momentum_price=float(momentum[vector_index]),
            atr_price=float(atr[vector_index]),
            observed_spread_price=float(spread[vector_index]),
            tp_distance_price=float(max(0.5 * atr[vector_index], 1.5 * spread[vector_index])),
            direction_correct_at_six_bars=direction_correct,
        ))
    return signals


def _entry_fill(arm: str, signal: Signal, bar: Bar) -> tuple[float, float, float, float]:
    pip = pip_size(signal.pair)
    mid = bar.mid("o")
    if arm == "RAW_SIGNAL":
        return mid, 0.0, 0.0, 0.0
    if arm == "EXECUTABLE_BASE":
        price = bar.side("ask", "o") if signal.side == "LONG" else bar.side("bid", "o")
        spread_cost = abs(price - mid) / pip
        return price, spread_cost, 0.0, 0.0
    base = bar.side("ask", "o") if signal.side == "LONG" else bar.side("bid", "o")
    extreme = bar.side("ask", "h") if signal.side == "LONG" else bar.side("bid", "l")
    price = extreme + pip * 0.3 if signal.side == "LONG" else extreme - pip * 0.3
    spread_cost = abs(base - mid) / pip
    latency_cost = abs(extreme - base) / pip
    return price, spread_cost, 0.3, latency_cost


def _exit_price(arm: str, position: Position, bar: Bar, field: str) -> tuple[float, float, float]:
    side = position.signal.side
    pip = pip_size(position.signal.pair)
    mid = bar.mid(field)
    if arm == "RAW_SIGNAL":
        return mid, 0.0, 0.0
    executable = bar.side("bid", field) if side == "LONG" else bar.side("ask", field)
    spread_cost = abs(executable - mid) / pip
    slippage = 0.3 if arm == "ADVERSE_STRESS" else 0.0
    price = executable - pip * slippage if side == "LONG" else executable + pip * slippage
    return price, spread_cost, slippage


def _conversion(
    pair: str,
    quote_pnl: float,
    source_time: int,
    field: str,
    usd_jpy: dict[int, Bar],
) -> tuple[float | None, float | None, str]:
    if pair.endswith("_JPY"):
        return quote_pnl, 1.0, "JPY_QUOTE_DIRECT"
    cross = usd_jpy.get(source_time)
    if cross is None:
        return None, None, "MISSING_EXACT_USD_JPY"
    side = "bid" if quote_pnl >= 0 else "ask"
    rate = cross.side(side, field)
    return quote_pnl * rate, rate, f"USD_JPY_{side.upper()}_{field.upper()}"


def _close_position(
    state: ArmState,
    pair: str,
    bar: Bar,
    reason: str,
    field: str,
    usd_jpy: dict[int, Bar],
    tp_fill: bool = False,
) -> None:
    position = state.positions.pop(pair)
    signal = position.signal
    if tp_fill:
        exit_price = position.tp_price
        exit_spread = None
        exit_slippage = 0.3 if state.arm == "ADVERSE_STRESS" else 0.0
        pip = pip_size(pair)
        if state.arm == "ADVERSE_STRESS":
            exit_price += -pip * exit_slippage if signal.side == "LONG" else pip * exit_slippage
        conversion_field = "o"
    else:
        exit_price, exit_spread, exit_slippage = _exit_price(
            state.arm, position, bar, field
        )
        conversion_field = field
    net_pips = signed_pips(pair, signal.side, position.entry_price, exit_price)
    quote_pnl = 1000.0 * (exit_price - position.entry_price) * (
        1.0 if signal.side == "LONG" else -1.0
    )
    jpy, rate, conversion_source = _conversion(
        pair, quote_pnl, bar.time, conversion_field, usd_jpy
    )
    if jpy is None:
        state.missing_jpy_conversions += 1
    else:
        state.realized_jpy += jpy
    if tp_fill and state.arm != "RAW_SIGNAL":
        # BID/ASK OHLC proves the executable TP touch but does not reveal the
        # exact simultaneous opposite side needed to reconstruct intrabar mid.
        # Leave the per-trade gross comparator null rather than fabricate it.
        gross_close_mark = None
        gross_comparator_basis = "UNAVAILABLE_INTRABAR_TP_MID"
    else:
        gross_close_mark = signed_pips(
            pair, signal.side, position.entry_mid,
            position.tp_price if tp_fill else bar.mid(field),
        )
        gross_comparator_basis = (
            "RAW_TP_TARGET" if tp_fill else f"OBSERVED_MID_{field.upper()}"
        )
    row = {
        "schema": "QR_M5_EMA_STATE_IMPULSE_TRADE_V1",
        "signal_id": signal.signal_id,
        "pair": pair,
        "side": signal.side,
        "arm": state.arm,
        "decision_time": iso_utc(signal.decision_time),
        "entry_time": iso_utc(position.entry_time),
        "exit_time": iso_utc(
            bar.time + BAR_SECONDS if tp_fill or field == "c" else bar.time
        ),
        "exit_source_bar_time": iso_utc(bar.time),
        "exit_reason": reason,
        "entry_price": position.entry_price,
        "exit_price": exit_price,
        "tp_distance_price": signal.tp_distance_price,
        "net_pips": net_pips,
        "gross_close_mark_pips": gross_close_mark,
        "gross_comparator_basis": gross_comparator_basis,
        "entry_spread_pips": position.entry_spread_pips,
        "known_exit_spread_pips": exit_spread,
        "entry_slippage_pips": position.entry_slippage_pips,
        "exit_slippage_pips": exit_slippage,
        "latency_proxy_pips": position.latency_proxy_pips,
        "mfe_pips": position.mfe_pips,
        "mae_pips": position.mae_pips,
        "age_bars": position.bars_observed,
        "quote_pnl": quote_pnl,
        "pnl_jpy": jpy,
        "jpy_conversion_rate": rate,
        "jpy_conversion_source": conversion_source,
        "terminal_liquidation": reason in {
            "SPLIT_TERMINAL_LIQUIDATION", "DATA_GAP_TERMINAL_LIQUIDATION"
        },
    }
    state.trades.append(row)
    outcome = state.outcomes[signal.signal_id]
    outcome.update({"status": "CLOSED", "exit_reason": reason,
                    "entry_time": row["entry_time"], "exit_time": row["exit_time"]})


def _update_tp_and_excursion(state: ArmState, pair: str, bar: Bar, usd_jpy: dict[int, Bar]) -> None:
    position = state.positions.get(pair)
    if position is None:
        return
    side = position.signal.side
    if state.arm == "RAW_SIGNAL":
        favorable = bar.mid("h") if side == "LONG" else bar.mid("l")
        adverse = bar.mid("l") if side == "LONG" else bar.mid("h")
    else:
        favorable = bar.side("bid", "h") if side == "LONG" else bar.side("ask", "l")
        adverse = bar.side("bid", "l") if side == "LONG" else bar.side("ask", "h")
    if state.arm == "ADVERSE_STRESS" and bar.time == position.entry_time:
        close, _, _ = _exit_price(state.arm, position, bar, "c")
        value = signed_pips(pair, side, position.entry_price, close)
        position.mfe_pips = max(position.mfe_pips, 0.0)
        position.mae_pips = min(position.mae_pips, value, 0.0)
        position.bars_observed += 1
        return
    position.mfe_pips = max(
        position.mfe_pips, signed_pips(pair, side, position.entry_price, favorable)
    )
    position.mae_pips = min(
        position.mae_pips, signed_pips(pair, side, position.entry_price, adverse)
    )
    position.bars_observed += 1
    hit = favorable >= position.tp_price if side == "LONG" else favorable <= position.tp_price
    if hit:
        _close_position(state, pair, bar, "TP", "o", usd_jpy, tp_fill=True)


def _mark_equity(
    state: ArmState,
    when: int,
    latest: dict[str, Bar],
    usd_jpy: dict[int, Bar],
    initial: float,
) -> None:
    unrealized = 0.0
    missing = False
    for pair, position in state.positions.items():
        bar = latest.get(pair)
        if bar is None:
            continue
        exit_price, _, _ = _exit_price(state.arm, position, bar, "c")
        quote = 1000.0 * (exit_price - position.entry_price) * (
            1.0 if position.signal.side == "LONG" else -1.0
        )
        converted, _, _ = _conversion(pair, quote, bar.time, "c", usd_jpy)
        if converted is None:
            missing = True
        else:
            unrealized += converted
    if not missing:
        state.equity_marks.append((when, initial + state.realized_jpy + unrealized))


def simulate_arm(
    arm: str,
    bars_by_pair: dict[str, list[Bar]],
    signals: list[Signal],
    phase_start: int,
    phase_end: int,
    prereg: dict,
) -> ArmState:
    state = ArmState(arm)
    bar_maps = {pair: {bar.time: bar for bar in rows} for pair, rows in bars_by_pair.items()}
    usd_jpy = bar_maps["USD_JPY"]
    signals_at: dict[int, list[Signal]] = defaultdict(list)
    priority = {pair: index for index, pair in enumerate(prereg["strategy"][
        "pair_priority_for_simultaneous_signals"
    ])}
    for signal in signals:
        signals_at[signal.decision_time].append(signal)
        state.outcomes[signal.signal_id] = {"status": "RAW_RECORDED"}
    for values in signals_at.values():
        values.sort(key=lambda signal: (priority[signal.pair], signal.signal_id))
    times = sorted({
        bar.time for rows in bars_by_pair.values() for bar in rows
        if phase_start <= bar.time < phase_end
    } | set(signals_at))
    latest: dict[str, Bar] = {}
    previous: dict[str, Bar] = {}
    gap_halted: set[str] = set()
    for when in times:
        current = {pair: mapping[when] for pair, mapping in bar_maps.items() if when in mapping}
        # If another symbol advances the common clock while this symbol's exact
        # next M5 row is absent, the gap is already observable.  Liquidate at
        # the last executable close immediately; do not wait for this pair to
        # resume and consume portfolio capacity with stale inventory.
        for pair, prior in list(previous.items()):
            if (
                pair not in current
                and pair not in gap_halted
                and when >= prior.time + BAR_SECONDS
            ):
                if pair in state.positions:
                    _close_position(
                        state, pair, prior, "DATA_GAP_TERMINAL_LIQUIDATION", "c", usd_jpy
                    )
                pending = state.pending.pop(pair, None)
                if pending is not None:
                    state.outcomes[pending.signal.signal_id]["status"] = "GAP_EXPIRED_NO_FILL"
                gap_halted.add(pair)
        for pair, bar in current.items():
            prior = previous.get(pair)
            if (
                prior is not None
                and bar.time - prior.time != BAR_SECONDS
                and pair not in gap_halted
            ):
                if pair in state.positions:
                    _close_position(
                        state, pair, prior, "DATA_GAP_TERMINAL_LIQUIDATION", "c", usd_jpy
                    )
                pending = state.pending.pop(pair, None)
                if pending is not None:
                    state.outcomes[pending.signal.signal_id]["status"] = "GAP_EXPIRED_NO_FILL"
            gap_halted.discard(pair)
            previous[pair] = bar
            latest[pair] = bar
        for pair, pending in list(state.pending.items()):
            if pending.due_time < when:
                state.pending.pop(pair)
                state.outcomes[pending.signal.signal_id]["status"] = "MISSING_STRICT_FILL"
        for pair, position in list(state.positions.items()):
            bar = current.get(pair)
            if bar is not None and position.max_age_due_time == when:
                _close_position(state, pair, bar, "MAX_AGE", "o", usd_jpy)
        for pair, pending in list(state.pending.items()):
            if pending.due_time != when:
                continue
            bar = current.get(pair)
            state.pending.pop(pair)
            if bar is None:
                state.outcomes[pending.signal.signal_id]["status"] = "MISSING_STRICT_FILL"
                continue
            price, spread_cost, slip, latency = _entry_fill(arm, pending.signal, bar)
            tp = price + pending.signal.tp_distance_price * (
                1.0 if pending.signal.side == "LONG" else -1.0
            )
            state.positions[pair] = Position(
                signal=pending.signal, arm=arm, entry_time=when,
                entry_price=price, entry_mid=bar.mid("o"), tp_price=tp,
                max_age_due_time=when + prereg["strategy"]["max_age_bars"] * BAR_SECONDS,
                entry_spread_pips=spread_cost, entry_slippage_pips=slip,
                latency_proxy_pips=latency,
            )
            state.outcomes[pending.signal.signal_id]["status"] = "OPEN"
        for signal in signals_at.get(when, []):
            if signal.pair in gap_halted:
                # Preserve the cost-independent RAW proposal in outcomes, but
                # never turn a decision at the first missing M5 boundary into
                # an expected order that can fill when the pair later resumes.
                state.outcomes[signal.signal_id]["status"] = "GAP_HALTED_NO_ORDER"
                continue
            pair_busy = signal.pair in state.positions or signal.pair in state.pending
            total_reserved = len(state.positions) + len(state.pending)
            if pair_busy or total_reserved >= prereg["strategy"][
                "hard_max_open_positions_total"
            ]:
                state.outcomes[signal.signal_id]["status"] = "CAPACITY_BLOCKED"
            else:
                state.pending[signal.pair] = Pending(signal, signal.expected_fill_time)
                state.outcomes[signal.signal_id]["status"] = "PENDING"
        for pair, bar in current.items():
            _update_tp_and_excursion(state, pair, bar, usd_jpy)
        state.inventory_samples.append(len(state.positions))
        _mark_equity(
            state, when + BAR_SECONDS, latest, usd_jpy,
            prereg["reporting"]["initial_equity_jpy"],
        )
    for pair, pending in list(state.pending.items()):
        state.pending.pop(pair)
        state.outcomes[pending.signal.signal_id]["status"] = "SPLIT_END_EXPIRED_NO_FILL"
    for pair in list(state.positions):
        eligible = [bar for bar in bars_by_pair[pair] if bar.time < phase_end]
        if not eligible:
            raise ValueError("no terminal bar for open position")
        _close_position(
            state, pair, eligible[-1], "SPLIT_TERMINAL_LIQUIDATION", "c", usd_jpy
        )
    state.equity_marks.append((
        phase_end,
        prereg["reporting"]["initial_equity_jpy"] + state.realized_jpy,
    ))
    return state


def quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability,
                             method="linear"))


def block_bootstrap_lcb(trades: list[dict], resamples: int, seed: int) -> float | None:
    if not trades:
        return None
    sums: dict[str, float] = defaultdict(float)
    counts: Counter = Counter()
    for trade in trades:
        day = trade["decision_time"][:10]
        sums[day] += trade["net_pips"]
        counts[day] += 1
    first = min(dt.date.fromisoformat(day) for day in sums)
    last = max(dt.date.fromisoformat(day) for day in sums)
    days = []
    while first <= last:
        days.append(first.isoformat())
        first += dt.timedelta(days=1)
    day_sums = np.asarray([sums.get(day, 0.0) for day in days], dtype=np.float64)
    day_counts = np.asarray([counts.get(day, 0) for day in days], dtype=np.float64)
    block = 5
    if len(days) < block:
        return None
    rng = np.random.default_rng(seed)
    blocks = int(math.ceil(len(days) / block))
    starts = rng.integers(0, len(days) - block + 1, size=(resamples, blocks))
    indices = (starts[:, :, None] + np.arange(block)).reshape(resamples, -1)[:, :len(days)]
    boot_sums = day_sums[indices].sum(axis=1)
    boot_counts = day_counts[indices].sum(axis=1)
    means = np.divide(boot_sums, boot_counts, out=np.full(resamples, np.nan),
                      where=boot_counts > 0)
    if not np.any(np.isfinite(means)):
        return None
    return float(np.nanquantile(means, 0.05, method="linear"))


def _calendar_months(start: int, end: int) -> list[str]:
    cursor = dt.datetime.fromtimestamp(start, UTC).date().replace(day=1)
    last = dt.datetime.fromtimestamp(end - 1, UTC).date().replace(day=1)
    output = []
    while cursor <= last:
        output.append(cursor.strftime("%Y-%m"))
        cursor = (
            cursor.replace(year=cursor.year + 1, month=1)
            if cursor.month == 12 else cursor.replace(month=cursor.month + 1)
        )
    return output


def summarize_arm(
    state: ArmState,
    signals: list[Signal],
    prereg: dict,
    phase_start: int,
    phase_end: int,
) -> dict:
    trades = state.trades
    pips = [trade["net_pips"] for trade in trades]
    pair_values: dict[str, list[float]] = defaultdict(list)
    month_values: dict[str, list[float]] = defaultdict(list)
    month_pnl_jpy: dict[str, float] = defaultdict(float)
    daily_counts: Counter = Counter()
    for trade in trades:
        pair_values[trade["pair"]].append(trade["net_pips"])
        month_values[trade["exit_time"][:7]].append(trade["net_pips"])
        if trade["pnl_jpy"] is not None:
            month_pnl_jpy[trade["exit_time"][:7]] += trade["pnl_jpy"]
        daily_counts[trade["decision_time"][:10]] += 1
    weights = list(daily_counts.values())
    n_eff = (sum(weights) ** 2 / sum(value * value for value in weights)) if weights else 0.0
    ages = [trade["age_bars"] for trade in trades]
    peak = prereg["reporting"]["initial_equity_jpy"]
    max_drawdown = 0.0
    for _, equity in state.equity_marks:
        peak = max(peak, equity)
        if peak > 0:
            max_drawdown = min(max_drawdown, equity / peak - 1.0)
    initial = prereg["reporting"]["initial_equity_jpy"]
    months = _calendar_months(phase_start, phase_end)
    monthly_multiples = {}
    month_start_equity = initial
    for month in months:
        pnl = month_pnl_jpy.get(month, 0.0)
        monthly_multiples[month] = (
            (month_start_equity + pnl) / month_start_equity
            if month_start_equity > 0 else None
        )
        month_start_equity += pnl
    pair_counts = Counter(trade["pair"] for trade in trades)
    tp_ages = [trade["age_bars"] for trade in trades if trade["exit_reason"] == "TP"]
    return {
        "proposals_received": len(signals),
        "proposals_per_active_utc_day": (
            len(signals) / len({signal.decision_time // 86400 for signal in signals})
            if signals else 0.0
        ),
        "cost_pass_count_without_entry_gate": len(signals),
        "filled_trades": len(trades),
        "turnover_base_units": 2 * 1000 * len(trades),
        "net_pips_per_proposal": sum(pips) / len(signals) if signals else None,
        "capacity_blocks": sum(
            outcome["status"] == "CAPACITY_BLOCKED" for outcome in state.outcomes.values()
        ),
        "missing_strict_fills": sum(outcome["status"] in {
            "MISSING_STRICT_FILL", "GAP_EXPIRED_NO_FILL", "SPLIT_END_EXPIRED_NO_FILL",
            "GAP_HALTED_NO_ORDER",
        } for outcome in state.outcomes.values()),
        "tp_exits": sum(trade["exit_reason"] == "TP" for trade in trades),
        "max_age_exits": sum(trade["exit_reason"] == "MAX_AGE" for trade in trades),
        "gap_terminal_exits": sum(
            trade["exit_reason"] == "DATA_GAP_TERMINAL_LIQUIDATION" for trade in trades
        ),
        "split_terminal_exits": sum(
            trade["exit_reason"] == "SPLIT_TERMINAL_LIQUIDATION" for trade in trades
        ),
        "terminal_liquidation_net_pips": sum(
            trade["net_pips"] for trade in trades if trade["terminal_liquidation"]
        ),
        "terminal_liquidation_pnl_jpy": sum(
            trade["pnl_jpy"] for trade in trades
            if trade["terminal_liquidation"] and trade["pnl_jpy"] is not None
        ),
        "mean_pips": statistics.mean(pips) if pips else None,
        "median_pips": statistics.median(pips) if pips else None,
        "block_bootstrap_95pct_lcb_pips": block_bootstrap_lcb(
            trades, prereg["statistics"]["bootstrap_resamples"],
            prereg["statistics"]["bootstrap_seed"],
        ),
        "pair_mean_pips": {
            pair: statistics.mean(pair_values[pair]) if pair_values[pair] else None
            for pair in prereg["input"]["symbols"]
        },
        "pair_trade_counts": {
            pair: pair_counts.get(pair, 0) for pair in prereg["input"]["symbols"]
        },
        "calendar_month_mean_pips": {
            month: statistics.mean(month_values[month]) if month_values[month] else None
            for month in months
        },
        "positive_pair_count": sum(
            bool(values) and statistics.mean(values) > 0 for values in pair_values.values()
        ),
        "positive_calendar_month_fraction": (
            sum(bool(month_values[month]) and statistics.mean(month_values[month]) > 0
                for month in months) / len(months) if months else 0.0
        ),
        "calendar_month_pnl_jpy": {
            month: month_pnl_jpy.get(month, 0.0) for month in months
        },
        "calendar_month_equity_multiples": monthly_multiples,
        "direction_accuracy_at_six_bars": (
            sum(signal.direction_correct_at_six_bars is True for signal in signals) /
            sum(signal.direction_correct_at_six_bars is not None for signal in signals)
            if any(signal.direction_correct_at_six_bars is not None for signal in signals)
            else None
        ),
        "mean_mfe_pips": statistics.mean(
            [trade["mfe_pips"] for trade in trades]
        ) if trades else None,
        "mean_mae_pips": statistics.mean(
            [trade["mae_pips"] for trade in trades]
        ) if trades else None,
        "tp_hit_age_q50_bars": quantile(tp_ages, 0.50),
        "tp_hit_age_q90_bars": quantile(tp_ages, 0.90),
        "mean_entry_spread_pips": statistics.mean(
            [trade["entry_spread_pips"] for trade in trades]
        ) if trades else None,
        "mean_explicit_slippage_pips": statistics.mean([
            trade["entry_slippage_pips"] + trade["exit_slippage_pips"]
            for trade in trades
        ]) if trades else None,
        "mean_latency_proxy_pips": statistics.mean(
            [trade["latency_proxy_pips"] for trade in trades]
        ) if trades else None,
        "mean_gross_close_mark_pips_when_observable": statistics.mean([
            trade["gross_close_mark_pips"] for trade in trades
            if trade["gross_close_mark_pips"] is not None
        ]) if any(trade["gross_close_mark_pips"] is not None for trade in trades) else None,
        "gross_close_mark_comparator_coverage": (
            sum(trade["gross_close_mark_pips"] is not None for trade in trades) /
            len(trades) if trades else 0.0
        ),
        "inventory_peak": max(state.inventory_samples, default=0),
        "inventory_count_q50": quantile(state.inventory_samples, 0.50),
        "inventory_count_q90": quantile(state.inventory_samples, 0.90),
        "inventory_count_q99": quantile(state.inventory_samples, 0.99),
        "inventory_age_q50_bars": quantile(ages, 0.50),
        "inventory_age_q90_bars": quantile(ages, 0.90),
        "inventory_age_q99_bars": quantile(ages, 0.99),
        "terminal_open_inventory": len(state.positions),
        "missing_jpy_conversions": state.missing_jpy_conversions,
        "realized_pnl_jpy": state.realized_jpy,
        "equity_multiple": (initial + state.realized_jpy) / initial,
        "max_drawdown": max_drawdown,
        "drawdown_basis": "completed-source-bar portfolio MTM",
        "active_utc_decision_days": len(daily_counts),
        "N_eff_utc_day_kish": n_eff,
        "nonfinite_accounting": any(
            not math.isfinite(value) for value in pips
        ),
    }


def phase_replay(
    prereg: dict,
    bars_by_pair: dict[str, list[Bar]],
    phase_name: str,
    prereg_canonical_sha: str,
) -> tuple[dict, list[Signal], dict[str, ArmState]]:
    split = prereg["splits"][phase_name]
    start, end = parse_time(split["from_utc"]), parse_time(split["to_utc"])
    signals = []
    for pair in prereg["input"]["symbols"]:
        signals.extend(build_signals(pair, bars_by_pair[pair], start, end,
                                     prereg_canonical_sha))
    signals.sort(key=lambda item: (item.decision_time, item.pair, item.signal_id))
    if len({signal.signal_id for signal in signals}) != len(signals):
        raise ValueError("duplicate signal_id")
    states = {
        arm: simulate_arm(arm, bars_by_pair, signals, start, end, prereg)
        for arm in ARMS
    }
    if any(set(state.outcomes) != {signal.signal_id for signal in signals}
           for state in states.values()):
        raise AssertionError("three arms did not receive identical signal IDs")
    summaries = {
        arm: summarize_arm(state, signals, prereg, start, end)
        for arm, state in states.items()
    }
    raw_mean = summaries["RAW_SIGNAL"]["mean_pips"]
    raw_per_proposal = summaries["RAW_SIGNAL"]["net_pips_per_proposal"]
    for arm in ("EXECUTABLE_BASE", "ADVERSE_STRESS"):
        value = summaries[arm]["mean_pips"]
        summaries[arm]["delta_vs_raw_mean_pips"] = (
            value - raw_mean if value is not None and raw_mean is not None else None
        )
        summaries[arm]["total_drag_vs_raw_pips_per_trade"] = (
            raw_mean - value if value is not None and raw_mean is not None else None
        )
        per_proposal = summaries[arm]["net_pips_per_proposal"]
        summaries[arm]["total_drag_vs_raw_pips_per_proposal"] = (
            raw_per_proposal - per_proposal
            if raw_per_proposal is not None and per_proposal is not None else None
        )
    for arm in ARMS:
        summaries[arm]["break_even_roundtrip_cost_pips"] = raw_mean
    return {
        "phase": phase_name,
        "raw_proposals": len(signals),
        "signal_id_set_sha256": sha_bytes(canonical(sorted(
            signal.signal_id for signal in signals
        ))),
        "cost_decomposition": {
            "gross_RAW_SIGNAL_mean_pips": raw_mean,
            "gross_RAW_SIGNAL_pips_per_shared_proposal": raw_per_proposal,
            "observed_EXECUTABLE_BASE_mean_pips": summaries[
                "EXECUTABLE_BASE"
            ]["mean_pips"],
            "stressed_ADVERSE_STRESS_mean_pips": summaries[
                "ADVERSE_STRESS"
            ]["mean_pips"],
            "base_drag_vs_raw_pips_per_shared_proposal": summaries[
                "EXECUTABLE_BASE"
            ]["total_drag_vs_raw_pips_per_proposal"],
            "adverse_drag_vs_raw_pips_per_shared_proposal": summaries[
                "ADVERSE_STRESS"
            ]["total_drag_vs_raw_pips_per_proposal"],
            "note": "arm deltas include observed spread, declared slippage, latency-envelope and any arm-specific TP timing interaction; RAW proposals are never cost-gated",
        },
        "arms": summaries,
    }, signals, states


def pass_fail(prereg: dict, discovery: dict, validation: dict) -> dict:
    floor = prereg["pass_fail"]["discovery_density"]
    raw_d = discovery["arms"]["RAW_SIGNAL"]
    adverse_v = validation["arms"]["ADVERSE_STRESS"]
    density = (
        raw_d["filled_trades"] >= floor["executed_trades_gte"]
        and raw_d["active_utc_decision_days"] >= floor["active_utc_decision_days_gte"]
        and raw_d.get("pairs_with_50_trades", 0) >= floor["pairs_with_50_trades_gte"]
    )
    checks = {
        "discovery_total_density": density,
        "discovery_raw_mean_positive": raw_d["mean_pips"] is not None
        and raw_d["mean_pips"] > 0,
        "discovery_raw_lcb_positive": raw_d["block_bootstrap_95pct_lcb_pips"] is not None
        and raw_d["block_bootstrap_95pct_lcb_pips"] > 0,
        "validation_raw_mean_positive": validation["arms"]["RAW_SIGNAL"]["mean_pips"] is not None
        and validation["arms"]["RAW_SIGNAL"]["mean_pips"] > 0,
        "validation_base_mean_positive": validation["arms"]["EXECUTABLE_BASE"]["mean_pips"] is not None
        and validation["arms"]["EXECUTABLE_BASE"]["mean_pips"] > 0,
        "validation_adverse_mean_positive": adverse_v["mean_pips"] is not None
        and adverse_v["mean_pips"] > 0,
        "validation_adverse_lcb_positive": adverse_v["block_bootstrap_95pct_lcb_pips"] is not None
        and adverse_v["block_bootstrap_95pct_lcb_pips"] > 0,
        "validation_positive_pairs": adverse_v["positive_pair_count"] >= 2,
        "validation_positive_months": adverse_v["positive_calendar_month_fraction"]
        >= 2.0 / 3.0,
        "conversion_complete": all(
            validation["arms"][arm]["missing_jpy_conversions"] == 0 for arm in ARMS
        ),
        "terminal_inventory_zero": all(
            validation["arms"][arm]["terminal_open_inventory"] == 0 for arm in ARMS
        ),
        "finite_accounting": all(
            not validation["arms"][arm]["nonfinite_accounting"] for arm in ARMS
        ),
    }
    return {"checks": checks, "passed": all(checks.values())}


def _pair_trade_density(state: ArmState, minimum: int) -> int:
    counts = Counter(trade["pair"] for trade in state.trades)
    return sum(value >= minimum for value in counts.values())


def run() -> dict:
    prereg = load_preregistration()
    validation_checks = validate_preregistration(prereg)
    local_audit = audit_local_contracts(prereg)
    review = validate_review_receipt(prereg)
    if any(path.exists() for path in OUTPUTS):
        raise RuntimeError("ONE_SHOT_RESULT_ALREADY_EXISTS")
    audit = {
        **local_audit,
        "pairs": {},
        "review_receipt_sha256": sha_file(REVIEW_PATH),
        "validation_rows_decoded_before_discovery_lock": 0,
        "post_boundary_bytes_read": 0,
        "post_boundary_rows_decoded": 0,
        "post_boundary_labels_computed": 0,
    }
    prereg_canonical_sha = sha_bytes(canonical(prereg))
    discovery_rows = load_discovery_bytes(prereg, audit)
    calibration, calibration_signals, calibration_states = phase_replay(
        prereg, discovery_rows, "calibration", prereg_canonical_sha
    )
    discovery, discovery_signals, discovery_states = phase_replay(
        prereg, discovery_rows, "discovery", prereg_canonical_sha
    )
    discovery_pair_floor = _pair_trade_density(discovery_states["RAW_SIGNAL"], 50)
    discovery["arms"]["RAW_SIGNAL"]["pairs_with_50_trades"] = discovery_pair_floor
    discovery_lock = sha_bytes(canonical({
        "preregistration_canonical_sha256": prereg_canonical_sha,
        "fixed_strategy": prereg["strategy"],
        "discovery_signal_set_sha256": discovery["signal_id_set_sha256"],
        "discovery_summary": discovery,
        "no_configuration_selected": True,
    }))
    validation_only = load_validation_bytes(prereg, audit, discovery_lock)
    combined = {
        pair: discovery_rows[pair] + validation_only[pair]
        for pair in prereg["input"]["symbols"]
    }
    validation, validation_signals, validation_states = phase_replay(
        prereg, combined, "locked_validation", prereg_canonical_sha
    )
    validation["arms"]["RAW_SIGNAL"]["pairs_with_50_trades"] = _pair_trade_density(
        validation_states["RAW_SIGNAL"], 50
    )
    decision = pass_fail(prereg, discovery, validation)
    status = "INTERNAL_HISTORICAL_GATE_PASS" if decision["passed"] else "INTERNAL_HISTORICAL_GATE_FAIL"
    all_states = {
        phase: states for phase, states in (
            ("calibration", calibration_states),
            ("discovery", discovery_states),
            ("locked_validation", validation_states),
        )
    }
    signal_rows = []
    for phase_name, signals_for_phase in (
        ("calibration", calibration_signals),
        ("discovery", discovery_signals),
        ("locked_validation", validation_signals),
    ):
        states = all_states[phase_name]
        for signal in signals_for_phase:
            row = asdict(signal)
            row.update({
                "schema": "QR_M5_EMA_STATE_IMPULSE_SIGNAL_V1",
                "phase": phase_name,
                "decision_time": iso_utc(signal.decision_time),
                "decision_bar_time": iso_utc(signal.decision_bar_time),
                "expected_fill_time": iso_utc(signal.expected_fill_time),
                "entry_cost_gate_used": False,
                "arm_outcomes": {
                    arm: states[arm].outcomes[signal.signal_id] for arm in ARMS
                },
            })
            signal_rows.append(row)
    trade_rows = []
    for phase_name, states in all_states.items():
        for arm in ARMS:
            for trade in states[arm].trades:
                trade_rows.append({**trade, "phase": phase_name})
    trade_rows.sort(key=lambda row: (row["exit_time"], row["arm"], row["pair"], row["signal_id"]))
    result = {
        "schema": "QR_M5_EMA_STATE_IMPULSE_RESULT_V1",
        "candidate_id": prereg["candidate_id"],
        "status": status,
        "historical_gate_passed": decision["passed"],
        "profit_proven": False,
        "strategy_admitted": False,
        "live_order_authority": False,
        "preregistration_checks": validation_checks,
        "review": {
            "reviewer_task_id": review["reviewer_task_id"],
            "receipt_sha256": sha_file(REVIEW_PATH),
        },
        "hashes": {
            "preregistration_file_sha256": sha_file(PREREG_PATH),
            "preregistration_canonical_sha256": prereg_canonical_sha,
            "runner_sha256": sha_file(Path(__file__).resolve()),
            "dataset_sha256": prereg["input"]["canonical_dataset_sha256"],
            "discovery_lock_sha256": discovery_lock,
        },
        "audit": audit,
        "calibration": calibration,
        "discovery": discovery,
        "locked_validation": validation,
        "pass_fail": decision,
        "signal_rows": len(signal_rows),
        "trade_rows": len(trade_rows),
        "authority": {
            "network_attempts": 0,
            "credential_reads": 0,
            "external_order_attempts": 0,
            "external_orders": 0,
            "launchd_actions": 0,
            "git_actions": 0,
        },
        "limitations": prereg["limitations"],
    }
    result["result_sha256"] = sha_bytes(canonical(result))
    packet = {
        "schema": "QR_M5_EMA_STATE_IMPULSE_EVIDENCE_PACKET_V1",
        "candidate_id": prereg["candidate_id"],
        "status": status,
        "historical_gate_passed": decision["passed"],
        "profit_proven": False,
        "strategy_admitted": False,
        "result_sha256": result["result_sha256"],
        "preregistration_file_sha256": sha_file(PREREG_PATH),
        "runner_sha256": sha_file(Path(__file__).resolve()),
        "signals_ledger_rows": len(signal_rows),
        "trades_ledger_rows": len(trade_rows),
        "post_boundary_rows_decoded": 0,
        "external_orders": 0,
    }
    packet["packet_sha256"] = sha_bytes(canonical(packet))
    atomic_jsonl(SIGNALS_PATH, signal_rows)
    atomic_jsonl(TRADES_PATH, trade_rows)
    atomic_json(RESULT_PATH, result)
    atomic_json(PACKET_PATH, packet)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audit-only", action="store_true")
    mode.add_argument("--run-once", action="store_true")
    args = parser.parse_args()
    prereg = load_preregistration()
    checks = validate_preregistration(prereg)
    audit = audit_local_contracts(prereg)
    if args.audit_only:
        print(json.dumps({
            "status": "PREREGISTERED_AWAITING_INDEPENDENT_REVIEW",
            "preregistration_checks": len(checks),
            **audit,
        }, sort_keys=True))
        return 0
    result = run()
    print(json.dumps({
        "status": result["status"],
        "historical_gate_passed": result["historical_gate_passed"],
        "result_sha256": result["result_sha256"],
        "external_orders": 0,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
