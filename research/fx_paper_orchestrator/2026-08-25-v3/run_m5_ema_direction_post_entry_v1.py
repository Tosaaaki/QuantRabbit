"""Paper-only historical challenger for natural M5 EMA direction proposals.

The module is intentionally standalone and uses only the Python standard
library.  It never imports the live runtime, broker clients, credentials,
orders, launchd helpers, the R5 oracle, or either untracked V3 draft.

The checked-in preregistration is the authority for one bounded family:
one EMA direction-state proposal per completed M5 bar after warm-up, fanned
out to twelve frozen post-entry configurations and three common cost arms.
All output remains UNADMITTED_CHALLENGER evidence.
"""

from __future__ import annotations

import argparse
import calendar
import gzip
import hashlib
import json
import math
import os
import re
import stat
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING, ROUND_HALF_EVEN, localcontext
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


CANDIDATE_ID = "M5_EMA_DIRECTION_POST_ENTRY_V1"
PREREG_NAME = "M5_EMA_DIRECTION_POST_ENTRY_V1_PREREGISTRATION.json"
RESULT_NAME = "result_m5_ema_direction_post_entry_v1.json"
SOURCE_MANIFEST_NAME = "source_manifest.json"
SIGNAL_LEDGER_NAME = "raw_signal_ledger.jsonl"
ARTIFACT_MANIFEST_NAME = "artifact_manifest.json"

BAR_NS = 300_000_000_000
ZERO_SHA256 = "0" * 64
PRICE_KEYS = frozenset({"o", "h", "l", "c"})
ROW_KEYS = frozenset(
    {"ask", "bid", "complete", "granularity", "pair", "price", "time", "volume"}
)
TS_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})T(?P<clock>\d{2}:\d{2}:\d{2})\.(?P<fraction>\d{9})Z$"
)

D = Decimal
JPY_QUANTUM = D("0.000001")
RATIO_QUANTUM = D("0.000000000001")
PIP_QUANTUM = D("0.000001")

INSTRUMENTS: dict[str, dict[str, Decimal | int]] = {
    "EUR_USD": {"price_scale": 100_000, "pip": D("0.0001"), "tick": D("0.00001")},
    "USD_JPY": {"price_scale": 1_000, "pip": D("0.01"), "tick": D("0.001")},
    "AUD_USD": {"price_scale": 100_000, "pip": D("0.0001"), "tick": D("0.00001")},
}

COST_ARMS: dict[str, dict[str, Decimal | str]] = {
    "RAW_SIGNAL": {
        "slippage_pips_per_side": D("0"),
        "commission_bps_per_side": D("0"),
        "financing_bps_per_day": D("0"),
        "price_basis": "MID",
    },
    "EXECUTABLE_BASE": {
        "slippage_pips_per_side": D("0.3"),
        "commission_bps_per_side": D("0"),
        "financing_bps_per_day": D("0.5"),
        "price_basis": "BID_ASK",
    },
    "ADVERSE_STRESS": {
        "slippage_pips_per_side": D("0.9"),
        "commission_bps_per_side": D("0.2"),
        "financing_bps_per_day": D("1.5"),
        "price_basis": "BID_ASK",
    },
}

POLICIES = (
    "A_MAX_AGE_ONLY",
    "B_OPPOSITE_SIGNAL_OLDEST_FIRST",
    "C_TRAINING_MFE_Q40_TP",
    "D_TP_Q40_PROFIT_GIVEBACK",
)
MAX_AGES = (6, 12, 24)
CONFIG_IDS = tuple(f"{policy}__H{age:02d}" for policy in POLICIES for age in MAX_AGES)


class ChallengerError(RuntimeError):
    """A fail-closed contract or evidence error."""


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def embedded_sha256(value: Mapping[str, Any], field_name: str) -> str:
    body = dict(value)
    body.pop(field_name, None)
    return sha256_bytes(canonical_json_bytes(body))


def exact_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ChallengerError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value: str) -> None:
    raise ChallengerError(f"non-finite JSON constant: {value}")


def strict_json_loads(raw: str) -> Any:
    try:
        return json.loads(
            raw,
            object_pairs_hook=exact_object,
            parse_float=Decimal,
            parse_int=int,
            parse_constant=reject_constant,
        )
    except ChallengerError:
        raise
    except Exception as exc:  # pragma: no cover - defensive wrapper
        raise ChallengerError(f"invalid JSON: {exc}") from exc


def strict_json_file(path: Path) -> dict[str, Any]:
    value = strict_json_loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ChallengerError(f"expected JSON object: {path}")
    return value


def decimal_text(value: Decimal, quantum: Decimal = RATIO_QUANTUM) -> str:
    if not value.is_finite():
        raise ChallengerError("non-finite Decimal output")
    with localcontext() as ctx:
        ctx.prec = 60
        return format(value.quantize(quantum, rounding=ROUND_HALF_EVEN), "f")


def parse_epoch_ns(value: str) -> int:
    match = TS_RE.fullmatch(value)
    if not match:
        raise ChallengerError(f"timestamp is not canonical 9-digit UTC RFC3339: {value}")
    base = datetime.strptime(
        f"{match.group('date')}T{match.group('clock')}", "%Y-%m-%dT%H:%M:%S"
    ).replace(tzinfo=timezone.utc)
    seconds = calendar.timegm(base.utctimetuple())
    return seconds * 1_000_000_000 + int(match.group("fraction"))


def format_epoch_ns(value: int) -> str:
    seconds, fraction = divmod(value, 1_000_000_000)
    base = datetime.fromtimestamp(seconds, tz=timezone.utc)
    return f"{base:%Y-%m-%dT%H:%M:%S}.{fraction:09d}Z"


def utc_day(value_ns: int) -> str:
    return datetime.fromtimestamp(value_ns // 1_000_000_000, tz=timezone.utc).strftime(
        "%Y-%m-%d"
    )


def utc_month(value_ns: int) -> str:
    return datetime.fromtimestamp(value_ns // 1_000_000_000, tz=timezone.utc).strftime(
        "%Y-%m"
    )


def read_regular_no_follow(path: Path) -> bytes:
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ChallengerError(f"source must be a regular non-symlink file: {path}")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode):
            raise ChallengerError(f"opened source is not regular: {path}")
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise ChallengerError(f"source inode changed before open: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(fd)
        if (opened.st_dev, opened.st_ino, opened.st_size) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
        ):
            raise ChallengerError(f"source changed during read: {path}")
        return b"".join(chunks)
    finally:
        os.close(fd)


def atomic_write(path: Path, raw: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise ChallengerError(f"refusing to overwrite artifact: {path}")
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(tmp, flags, 0o600)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise ChallengerError(f"short write: {tmp}")
            view = view[written:]
        os.fsync(fd)
    except BaseException:
        os.close(fd)
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise
    else:
        os.close(fd)
    os.replace(tmp, path)
    directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


@dataclass(frozen=True)
class Bar:
    pair: str
    ordinal: int
    start_ns: int
    time: str
    bid_o: Decimal
    bid_h: Decimal
    bid_l: Decimal
    bid_c: Decimal
    ask_o: Decimal
    ask_h: Decimal
    ask_l: Decimal
    ask_c: Decimal
    row_sha256: str

    @property
    def mid_o(self) -> Decimal:
        return (self.bid_o + self.ask_o) / D(2)

    @property
    def mid_h(self) -> Decimal:
        return (self.bid_h + self.ask_h) / D(2)

    @property
    def mid_l(self) -> Decimal:
        return (self.bid_l + self.ask_l) / D(2)

    @property
    def mid_c(self) -> Decimal:
        return (self.bid_c + self.ask_c) / D(2)

    @property
    def end_ns(self) -> int:
        return self.start_ns + BAR_NS


@dataclass(frozen=True)
class Signal:
    signal_id: str
    pair: str
    direction: int
    decision_bar_ordinal: int
    decision_source_time: str
    decision_ns: int
    fill_bar_ordinal: int | None
    fill_source_time: str | None
    source_row_sha256: str
    ema3: Decimal
    ema12: Decimal


def require_decimal(value: Any, label: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, Decimal)):
        raise ChallengerError(f"{label} must be a JSON number")
    result = D(value)
    if not result.is_finite() or result <= 0:
        raise ChallengerError(f"{label} must be finite and positive")
    return result


def load_bars(path: Path, pair: str, expected_sha256: str) -> tuple[list[Bar], dict[str, Any]]:
    compressed = read_regular_no_follow(path)
    actual_sha = sha256_bytes(compressed)
    if actual_sha != expected_sha256:
        raise ChallengerError(
            f"source SHA mismatch for {pair}: expected {expected_sha256}, got {actual_sha}"
        )
    try:
        decompressed = gzip.decompress(compressed)
    except Exception as exc:
        raise ChallengerError(f"invalid gzip source for {pair}: {exc}") from exc
    if not decompressed.endswith(b"\n"):
        raise ChallengerError(f"source must end with LF: {path}")
    bars: list[Bar] = []
    previous_ns: int | None = None
    schema_sha = sha256_bytes(b"ask,bid,complete,granularity,pair,price,time,volume|o,h,l,c")
    for line_number, raw_line in enumerate(decompressed.splitlines(), 1):
        try:
            text = raw_line.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ChallengerError(f"invalid UTF-8 at {path}:{line_number}") from exc
        row = strict_json_loads(text)
        if not isinstance(row, dict) or frozenset(row) != ROW_KEYS:
            raise ChallengerError(f"unexpected row schema at {path}:{line_number}")
        if row["complete"] is not True or row["granularity"] != "M5" or row["price"] != "BA":
            raise ChallengerError(f"non-completed M5 BA row at {path}:{line_number}")
        if row["pair"] != pair:
            raise ChallengerError(f"pair mismatch at {path}:{line_number}")
        start_ns = parse_epoch_ns(str(row["time"]))
        if start_ns % BAR_NS != 0:
            raise ChallengerError(f"off-grid M5 timestamp at {path}:{line_number}")
        if previous_ns is not None and start_ns <= previous_ns:
            raise ChallengerError(f"timestamp reversal/duplicate at {path}:{line_number}")
        previous_ns = start_ns
        bid = row["bid"]
        ask = row["ask"]
        if not isinstance(bid, dict) or not isinstance(ask, dict):
            raise ChallengerError(f"BID/ASK must be objects at {path}:{line_number}")
        if frozenset(bid) != PRICE_KEYS or frozenset(ask) != PRICE_KEYS:
            raise ChallengerError(f"unexpected BID/ASK schema at {path}:{line_number}")
        b = {key: require_decimal(bid[key], f"bid.{key}") for key in PRICE_KEYS}
        a = {key: require_decimal(ask[key], f"ask.{key}") for key in PRICE_KEYS}
        if not (b["l"] <= b["o"] <= b["h"] and b["l"] <= b["c"] <= b["h"]):
            raise ChallengerError(f"invalid BID OHLC at {path}:{line_number}")
        if not (a["l"] <= a["o"] <= a["h"] and a["l"] <= a["c"] <= a["h"]):
            raise ChallengerError(f"invalid ASK OHLC at {path}:{line_number}")
        if any(a[key] < b[key] for key in PRICE_KEYS):
            raise ChallengerError(f"spread inversion at {path}:{line_number}")
        bars.append(
            Bar(
                pair=pair,
                ordinal=len(bars),
                start_ns=start_ns,
                time=str(row["time"]),
                bid_o=b["o"],
                bid_h=b["h"],
                bid_l=b["l"],
                bid_c=b["c"],
                ask_o=a["o"],
                ask_h=a["h"],
                ask_l=a["l"],
                ask_c=a["c"],
                row_sha256=sha256_bytes(raw_line),
            )
        )
    if len(bars) < 12:
        raise ChallengerError(f"source is shorter than EMA warmup: {path}")
    gaps = [
        (right.start_ns - left.start_ns) // 1_000_000_000
        for left, right in zip(bars, bars[1:])
        if right.start_ns - left.start_ns != BAR_NS
    ]
    return bars, {
        "pair": pair,
        "relative_or_absolute_path": str(path),
        "compressed_sha256": actual_sha,
        "compressed_size_bytes": len(compressed),
        "decompressed_sha256": sha256_bytes(decompressed),
        "decompressed_size_bytes": len(decompressed),
        "row_count": len(bars),
        "first_source_time": bars[0].time,
        "last_source_time": bars[-1].time,
        "timestamp_gap_count": len(gaps),
        "maximum_timestamp_gap_seconds": max(gaps, default=300),
        "schema_sha256": schema_sha,
        "integrity_error_count": 0,
        "arrival_sequence_heartbeat_fields_present": false_literal(),
    }


def false_literal() -> bool:
    """Makes security-significant false values visibly exact in call sites."""
    return False


def ema_signals(bars: Sequence[Bar], candidate_id: str = CANDIDATE_ID) -> list[Signal]:
    if len(bars) < 12:
        return []
    fast_seed: list[Decimal] = []
    slow_seed: list[Decimal] = []
    fast: Decimal | None = None
    slow: Decimal | None = None
    previous_direction: int | None = None
    result: list[Signal] = []
    alpha_fast = D(2) / D(4)
    alpha_slow = D(2) / D(13)
    with localcontext() as ctx:
        ctx.prec = 60
        for index, bar in enumerate(bars):
            close = bar.mid_c
            if fast is None:
                fast_seed.append(close)
                if len(fast_seed) == 3:
                    fast = sum(fast_seed, D(0)) / D(3)
            else:
                fast = alpha_fast * close + (D(1) - alpha_fast) * fast
            if slow is None:
                slow_seed.append(close)
                if len(slow_seed) == 12:
                    slow = sum(slow_seed, D(0)) / D(12)
            else:
                slow = alpha_slow * close + (D(1) - alpha_slow) * slow
            if fast is None or slow is None:
                continue
            delta = fast - slow
            if delta > 0:
                direction = 1
            elif delta < 0:
                direction = -1
            else:
                direction = previous_direction if previous_direction is not None else 1
            previous_direction = direction
            decision_ns = bar.end_ns
            fill_bar = bars[index + 1] if index + 1 < len(bars) else None
            fill_ordinal = fill_bar.ordinal if fill_bar is not None else None
            fill_time = fill_bar.time if fill_bar is not None else None
            material = f"{candidate_id}|{bar.pair}|{decision_ns}|{direction}".encode(
                "ascii"
            )
            result.append(
                Signal(
                    signal_id=sha256_bytes(material),
                    pair=bar.pair,
                    direction=direction,
                    decision_bar_ordinal=bar.ordinal,
                    decision_source_time=bar.time,
                    decision_ns=decision_ns,
                    fill_bar_ordinal=fill_ordinal,
                    fill_source_time=fill_time,
                    source_row_sha256=bar.row_sha256,
                    ema3=fast,
                    ema12=slow,
                )
            )
    return result


def nearest_rank(values: Sequence[Decimal], quantile: Decimal) -> Decimal:
    if not values or not (D(0) < quantile <= D(1)):
        raise ChallengerError("nearest-rank requires a nonempty sample and q in (0,1]")
    ordered = sorted(values)
    rank = int((quantile * D(len(ordered))).to_integral_value(rounding=ROUND_CEILING))
    return ordered[max(1, rank) - 1]


def executable_mfe_pips(
    bars: Sequence[Bar], fill_index: int, direction: int, max_age: int
) -> Decimal:
    if fill_index + max_age > len(bars):
        raise ChallengerError("full MFE horizon is unavailable")
    entry = bars[fill_index]
    pip = D(INSTRUMENTS[entry.pair]["pip"])
    path = bars[fill_index : fill_index + max_age]
    if direction > 0:
        return max((bar.bid_h - entry.ask_o) / pip for bar in path)
    return max((entry.bid_o - bar.ask_l) / pip for bar in path)


def freeze_mfe_targets(
    corpus: Mapping[str, Sequence[Bar]],
    signals: Mapping[str, Sequence[Signal]],
    tuning_start_ns: int,
    tuning_end_ns: int,
) -> dict[str, Any]:
    targets: dict[str, Any] = {}
    for pair in sorted(corpus):
        pair_targets: dict[str, Any] = {}
        tick_pips = D(INSTRUMENTS[pair]["tick"]) / D(INSTRUMENTS[pair]["pip"])
        for direction in (-1, 1):
            direction_key = "LONG" if direction > 0 else "SHORT"
            age_targets: dict[str, Any] = {}
            for max_age in MAX_AGES:
                samples: list[Decimal] = []
                for signal in signals[pair]:
                    if signal.direction != direction or signal.fill_bar_ordinal is None:
                        continue
                    if not (tuning_start_ns <= signal.decision_ns < tuning_end_ns):
                        continue
                    fill_index = signal.fill_bar_ordinal
                    if fill_index + max_age > len(corpus[pair]):
                        continue
                    final_bar = corpus[pair][fill_index + max_age - 1]
                    if final_bar.end_ns > tuning_end_ns:
                        continue
                    samples.append(executable_mfe_pips(corpus[pair], fill_index, direction, max_age))
                if not samples:
                    raise ChallengerError(f"empty tuning MFE stratum: {pair}/{direction_key}/{max_age}")
                raw_q40 = nearest_rank(samples, D("0.40"))
                frozen = max(raw_q40, tick_pips)
                age_targets[str(max_age)] = {
                    "sample_count": len(samples),
                    "raw_q40_pips": decimal_text(raw_q40, PIP_QUANTUM),
                    "frozen_target_pips": decimal_text(frozen, PIP_QUANTUM),
                    "minimum_tick_pips": decimal_text(tick_pips, PIP_QUANTUM),
                }
            pair_targets[direction_key] = age_targets
        targets[pair] = pair_targets
    body = {
        "schema_version": 1,
        "statistic": "nearest_rank_ceil_0.40",
        "cost_used": False,
        "walk_forward_used": False,
        "fallback_or_pooling_used": False,
        "strata": targets,
    }
    body["target_freeze_sha256"] = embedded_sha256(body, "target_freeze_sha256")
    return body


@dataclass
class Position:
    signal: Signal
    entry_bar_ordinal: int
    entry_ns: int
    entry_mid: Decimal
    entry_executable: Decimal
    target_pips: Decimal | None
    target_mid: Decimal | None
    target_executable: Decimal | None
    entry_prices: dict[str, Decimal]
    age_completed_bars: int = 0
    scheduled_exit_ordinal: int | None = None
    scheduled_exit_reason: str | None = None
    peak_progress_pips: Decimal = D("-Infinity")
    mfe_pips: Decimal = D("-Infinity")
    mae_pips: Decimal = D("Infinity")


@dataclass(frozen=True)
class Trade:
    signal_id: str
    pair: str
    direction: int
    decision_ns: int
    entry_ns: int
    exit_ns: int
    entry_bar_ordinal: int
    exit_bar_ordinal: int
    exit_reason: str
    age_completed_bars: int
    wall_age_seconds: int
    mfe_pips: Decimal
    mae_pips: Decimal
    pnl_jpy: dict[str, Decimal]
    return_ratio: dict[str, Decimal]


@dataclass
class PairRun:
    pair: str
    config_id: str
    signal_count: int
    fill_eligible_count: int
    filled_count: int
    cap_rejected_count: int
    no_future_fill_count: int
    max_open_lots: int
    terminal_liquidation_count: int
    terminal_open_inventory: int
    trades: list[Trade]
    mark_series: dict[str, list[tuple[int, Decimal]]]
    disposition_root_sha256: str
    filled_signal_id_set_sha256: str
    action_root_sha256: str


def config_parts(config_id: str) -> tuple[str, int]:
    if config_id not in CONFIG_IDS:
        raise ChallengerError(f"unknown config: {config_id}")
    policy, age_text = config_id.rsplit("__H", 1)
    return policy, int(age_text)


def target_for(
    target_freeze: Mapping[str, Any], pair: str, direction: int, max_age: int
) -> Decimal:
    direction_key = "LONG" if direction > 0 else "SHORT"
    value = target_freeze["strata"][pair][direction_key][str(max_age)][
        "frozen_target_pips"
    ]
    return D(str(value))


def entry_prices(bar: Bar, direction: int) -> dict[str, Decimal]:
    result: dict[str, Decimal] = {}
    pip = D(INSTRUMENTS[bar.pair]["pip"])
    for arm, policy in COST_ARMS.items():
        slip = D(policy["slippage_pips_per_side"]) * pip
        if arm == "RAW_SIGNAL":
            result[arm] = bar.mid_o
        elif direction > 0:
            result[arm] = bar.ask_o + slip
        else:
            result[arm] = bar.bid_o - slip
        if result[arm] <= 0:
            raise ChallengerError("entry price became nonpositive")
    return result


def open_position(
    signal: Signal,
    bar: Bar,
    target_pips: Decimal | None,
) -> Position:
    pip = D(INSTRUMENTS[bar.pair]["pip"])
    entry_exec = bar.ask_o if signal.direction > 0 else bar.bid_o
    entry_mid = bar.mid_o
    target_exec: Decimal | None = None
    target_mid: Decimal | None = None
    if target_pips is not None:
        distance = target_pips * pip
        if signal.direction > 0:
            target_exec = entry_exec + distance
            target_mid = entry_mid + distance
        else:
            target_exec = entry_exec - distance
            target_mid = entry_mid - distance
        if target_exec <= 0 or target_mid <= 0:
            raise ChallengerError("TP target became nonpositive")
    return Position(
        signal=signal,
        entry_bar_ordinal=bar.ordinal,
        entry_ns=bar.start_ns,
        entry_mid=entry_mid,
        entry_executable=entry_exec,
        target_pips=target_pips,
        target_mid=target_mid,
        target_executable=target_exec,
        entry_prices=entry_prices(bar, signal.direction),
    )


def common_exit_prices(
    position: Position,
    bar: Bar,
    exit_kind: str,
) -> tuple[Decimal, Decimal]:
    """Return the raw-mid and executable-side prices before cost-arm slippage."""
    if exit_kind == "TP":
        if position.target_mid is None or position.target_executable is None:
            raise ChallengerError("TP exit requested without a target")
        return position.target_mid, position.target_executable
    if exit_kind == "TERMINAL_CLOSE":
        mid = bar.mid_c
        executable = bar.bid_c if position.signal.direction > 0 else bar.ask_c
    else:
        mid = bar.mid_o
        executable = bar.bid_o if position.signal.direction > 0 else bar.ask_o
    return mid, executable


def pnl_for_exit(
    position: Position,
    bar: Bar,
    exit_kind: str,
    exit_ns: int,
    notional_jpy: Decimal,
) -> tuple[dict[str, Decimal], dict[str, Decimal]]:
    raw_exit, executable_exit = common_exit_prices(position, bar, exit_kind)
    direction = D(position.signal.direction)
    pip = D(INSTRUMENTS[bar.pair]["pip"])
    elapsed_seconds = D(max(0, exit_ns - position.entry_ns)) / D(1_000_000_000)
    elapsed_days = elapsed_seconds / D(86_400)
    pnl: dict[str, Decimal] = {}
    ratios: dict[str, Decimal] = {}
    with localcontext() as ctx:
        ctx.prec = 60
        for arm, policy in COST_ARMS.items():
            if arm == "RAW_SIGNAL":
                exit_price = raw_exit
            else:
                slip = D(policy["slippage_pips_per_side"]) * pip
                exit_price = (
                    executable_exit - slip
                    if position.signal.direction > 0
                    else executable_exit + slip
                )
            entry_price = position.entry_prices[arm]
            linear_return = direction * (exit_price - entry_price) / entry_price
            commission = D(2) * D(policy["commission_bps_per_side"]) / D(10_000)
            financing = D(policy["financing_bps_per_day"]) * elapsed_days / D(10_000)
            net_return = linear_return - commission - financing
            ratios[arm] = net_return
            pnl[arm] = (notional_jpy * net_return).quantize(
                JPY_QUANTUM, rounding=ROUND_HALF_EVEN
            )
    if pnl["ADVERSE_STRESS"] > pnl["EXECUTABLE_BASE"] + JPY_QUANTUM:
        raise ChallengerError("ADVERSE_STRESS was easier than EXECUTABLE_BASE")
    return pnl, ratios


def pnl_for_arm_at_mark(
    position: Position,
    bar: Bar,
    arm: str,
    mark_ns: int,
    notional_jpy: Decimal,
) -> Decimal:
    raw_exit, executable_exit = common_exit_prices(position, bar, "TERMINAL_CLOSE")
    policy = COST_ARMS[arm]
    pip = D(INSTRUMENTS[bar.pair]["pip"])
    if arm == "RAW_SIGNAL":
        exit_price = raw_exit
    else:
        slip = D(policy["slippage_pips_per_side"]) * pip
        exit_price = (
            executable_exit - slip
            if position.signal.direction > 0
            else executable_exit + slip
        )
    elapsed_seconds = D(max(0, mark_ns - position.entry_ns)) / D(1_000_000_000)
    elapsed_days = elapsed_seconds / D(86_400)
    direction = D(position.signal.direction)
    with localcontext() as ctx:
        ctx.prec = 60
        linear_return = (
            direction
            * (exit_price - position.entry_prices[arm])
            / position.entry_prices[arm]
        )
        commission = D(2) * D(policy["commission_bps_per_side"]) / D(10_000)
        financing = D(policy["financing_bps_per_day"]) * elapsed_days / D(10_000)
        return (notional_jpy * (linear_return - commission - financing)).quantize(
            JPY_QUANTUM, rounding=ROUND_HALF_EVEN
        )


def update_path_metrics(position: Position, bar: Bar) -> None:
    pip = D(INSTRUMENTS[bar.pair]["pip"])
    if position.signal.direction > 0:
        favorable = (bar.mid_h - position.entry_mid) / pip
        adverse = (bar.mid_l - position.entry_mid) / pip
    else:
        favorable = (position.entry_mid - bar.mid_l) / pip
        adverse = (position.entry_mid - bar.mid_h) / pip
    position.mfe_pips = max(position.mfe_pips, favorable)
    position.mae_pips = min(position.mae_pips, adverse)


def completed_close_progress_pips(position: Position, bar: Bar) -> Decimal:
    pip = D(INSTRUMENTS[bar.pair]["pip"])
    if position.signal.direction > 0:
        return (bar.bid_c - position.entry_executable) / pip
    return (position.entry_executable - bar.ask_c) / pip


def tp_touched(position: Position, bar: Bar) -> bool:
    target = position.target_executable
    if target is None:
        return False
    if position.signal.direction > 0:
        return bar.bid_h >= target
    return bar.ask_l <= target


def mark_position_pnl(
    position: Position,
    bar: Bar,
    arm: str,
    mark_ns: int,
    notional_jpy: Decimal,
) -> Decimal:
    return pnl_for_arm_at_mark(position, bar, arm, mark_ns, notional_jpy)


def chain_rows(rows: Iterable[Mapping[str, Any]]) -> tuple[str, int]:
    previous = ZERO_SHA256
    count = 0
    for row in rows:
        previous = sha256_bytes(bytes.fromhex(previous) + canonical_json_bytes(dict(row)))
        count += 1
    return previous, count


def signal_id_set_sha256(signal_ids: Iterable[str]) -> str:
    values = sorted(set(signal_ids))
    return sha256_bytes("".join(f"{value}\n" for value in values).encode("ascii"))


def simulate_pair(
    bars: Sequence[Bar],
    all_signals: Sequence[Signal],
    period_start_ns: int,
    period_end_ns: int,
    config_id: str,
    target_freeze: Mapping[str, Any],
    notional_jpy: Decimal,
    pair_open_lot_cap: int,
) -> PairRun:
    policy, max_age = config_parts(config_id)
    pair = bars[0].pair
    period_indices = [
        i for i, bar in enumerate(bars) if period_start_ns <= bar.start_ns < period_end_ns
    ]
    if not period_indices:
        raise ChallengerError(f"period has no bars for {pair}")
    last_index = period_indices[-1]
    period_signals = [
        signal
        for signal in all_signals
        if period_start_ns <= signal.decision_ns < period_end_ns
    ]
    signal_by_fill: dict[int, Signal] = {}
    dispositions: dict[str, str] = {}
    fill_eligible_count = 0
    for signal in period_signals:
        status = "NO_FUTURE_FILL_IN_PERIOD"
        if signal.fill_bar_ordinal is not None:
            fill_bar = bars[signal.fill_bar_ordinal]
            if (
                signal.fill_bar_ordinal > signal.decision_bar_ordinal
                and fill_bar.start_ns >= signal.decision_ns
                and fill_bar.start_ns < period_end_ns
            ):
                if signal.fill_bar_ordinal in signal_by_fill:
                    raise ChallengerError(f"duplicate fill ordinal for {pair}")
                signal_by_fill[signal.fill_bar_ordinal] = signal
                status = "PENDING_FILL"
                fill_eligible_count += 1
        dispositions[signal.signal_id] = status

    positions: list[Position] = []
    trades: list[Trade] = []
    realized = {arm: D(0) for arm in COST_ARMS}
    mark_series: dict[str, list[tuple[int, Decimal]]] = {arm: [] for arm in COST_ARMS}
    max_open_lots = 0
    cap_rejected = 0
    terminal_count = 0
    action_rows: list[dict[str, Any]] = []

    def close_one(position: Position, bar: Bar, reason: str, exit_kind: str, exit_ns: int) -> None:
        nonlocal terminal_count
        pnl, ratios = pnl_for_exit(position, bar, exit_kind, exit_ns, notional_jpy)
        for arm in COST_ARMS:
            realized[arm] += pnl[arm]
        wall_age = max(0, (exit_ns - position.entry_ns) // 1_000_000_000)
        trades.append(
            Trade(
                signal_id=position.signal.signal_id,
                pair=pair,
                direction=position.signal.direction,
                decision_ns=position.signal.decision_ns,
                entry_ns=position.entry_ns,
                exit_ns=exit_ns,
                entry_bar_ordinal=position.entry_bar_ordinal,
                exit_bar_ordinal=bar.ordinal,
                exit_reason=reason,
                age_completed_bars=position.age_completed_bars,
                wall_age_seconds=wall_age,
                mfe_pips=position.mfe_pips if position.mfe_pips.is_finite() else D(0),
                mae_pips=position.mae_pips if position.mae_pips.is_finite() else D(0),
                pnl_jpy=pnl,
                return_ratio=ratios,
            )
        )
        action_rows.append(
            {
                "signal_id": position.signal.signal_id,
                "pair": pair,
                "direction": position.signal.direction,
                "entry_bar_ordinal": position.entry_bar_ordinal,
                "exit_bar_ordinal": bar.ordinal,
                "exit_reason": reason,
                "exit_ns": exit_ns,
            }
        )
        if reason == "TERMINAL_LIQUIDATION":
            terminal_count += 1

    for index in period_indices:
        bar = bars[index]
        due = sorted(
            [position for position in positions if position.scheduled_exit_ordinal == index],
            key=lambda item: (item.entry_ns, item.signal.signal_id),
        )
        for position in due:
            if position in positions:
                close_one(
                    position,
                    bar,
                    position.scheduled_exit_reason or "SCHEDULED_EXIT",
                    "MARKET_OPEN",
                    bar.start_ns,
                )
                positions.remove(position)

        incoming = signal_by_fill.get(index)
        if incoming is not None:
            if policy == "B_OPPOSITE_SIGNAL_OLDEST_FIRST":
                opposite = sorted(
                    [p for p in positions if p.signal.direction == -incoming.direction],
                    key=lambda item: (item.entry_ns, item.signal.signal_id),
                )
                if opposite:
                    position = opposite[0]
                    close_one(position, bar, "OPPOSITE_SIGNAL_OLDEST_FIRST", "MARKET_OPEN", bar.start_ns)
                    positions.remove(position)
            if len(positions) >= pair_open_lot_cap:
                cap_rejected += 1
                dispositions[incoming.signal_id] = "CAP_REJECTED"
            else:
                target = (
                    target_for(target_freeze, pair, incoming.direction, max_age)
                    if policy in {
                        "C_TRAINING_MFE_Q40_TP",
                        "D_TP_Q40_PROFIT_GIVEBACK",
                    }
                    else None
                )
                position = open_position(incoming, bar, target)
                positions.append(position)
                dispositions[incoming.signal_id] = "FILLED"
                max_open_lots = max(max_open_lots, len(positions))
                action_rows.append(
                    {
                        "signal_id": incoming.signal_id,
                        "pair": pair,
                        "direction": incoming.direction,
                        "entry_bar_ordinal": bar.ordinal,
                        "entry_ns": bar.start_ns,
                        "action": "OPEN_FIXED_LOT",
                    }
                )

        for position in list(positions):
            update_path_metrics(position, bar)
            position.age_completed_bars = bar.ordinal - position.entry_bar_ordinal + 1
            if tp_touched(position, bar):
                close_one(position, bar, "FROZEN_TP_Q40", "TP", bar.end_ns)
                positions.remove(position)

        for position in positions:
            progress = completed_close_progress_pips(position, bar)
            position.peak_progress_pips = max(position.peak_progress_pips, progress)
            if index < last_index and position.scheduled_exit_ordinal is None:
                if position.age_completed_bars >= max_age:
                    position.scheduled_exit_ordinal = index + 1
                    position.scheduled_exit_reason = "FINITE_MAX_AGE"
                elif (
                    policy == "D_TP_Q40_PROFIT_GIVEBACK"
                    and position.target_pips is not None
                    and position.peak_progress_pips >= position.target_pips / D(2)
                    and progress <= position.peak_progress_pips / D(2)
                ):
                    position.scheduled_exit_ordinal = index + 1
                    position.scheduled_exit_reason = "PROFIT_GIVEBACK_UNWIND"

        if index == last_index:
            for position in sorted(
                list(positions), key=lambda item: (item.entry_ns, item.signal.signal_id)
            ):
                close_one(
                    position,
                    bar,
                    "TERMINAL_LIQUIDATION",
                    "TERMINAL_CLOSE",
                    bar.end_ns,
                )
                positions.remove(position)

        for arm in COST_ARMS:
            unrealized = sum(
                (mark_position_pnl(position, bar, arm, bar.end_ns, notional_jpy) for position in positions),
                D(0),
            )
            mark_series[arm].append((bar.end_ns, realized[arm] + unrealized))
        max_open_lots = max(max_open_lots, len(positions))

    ordered_dispositions = (
        {
            "signal_id": signal.signal_id,
            "pair": pair,
            "decision_ns": signal.decision_ns,
            "status": dispositions[signal.signal_id],
        }
        for signal in period_signals
    )
    disposition_root, disposition_count = chain_rows(ordered_dispositions)
    if disposition_count != len(period_signals):
        raise ChallengerError("disposition count mismatch")
    filled_ids = [signal_id for signal_id, status in dispositions.items() if status == "FILLED"]
    action_root, _ = chain_rows(action_rows)
    if positions:
        raise ChallengerError("terminal inventory was not liquidated")
    if max_open_lots > pair_open_lot_cap:
        raise ChallengerError("pair inventory cap exceeded")
    return PairRun(
        pair=pair,
        config_id=config_id,
        signal_count=len(period_signals),
        fill_eligible_count=fill_eligible_count,
        filled_count=len(filled_ids),
        cap_rejected_count=cap_rejected,
        no_future_fill_count=sum(
            status == "NO_FUTURE_FILL_IN_PERIOD" for status in dispositions.values()
        ),
        max_open_lots=max_open_lots,
        terminal_liquidation_count=terminal_count,
        terminal_open_inventory=0,
        trades=trades,
        mark_series=mark_series,
        disposition_root_sha256=disposition_root,
        filled_signal_id_set_sha256=signal_id_set_sha256(filled_ids),
        action_root_sha256=action_root,
    )


def percentile(values: Sequence[Decimal], q: Decimal) -> Decimal | None:
    if not values:
        return None
    return nearest_rank(values, q)


def safe_mean(values: Sequence[Decimal]) -> Decimal | None:
    return sum(values, D(0)) / D(len(values)) if values else None


def positive_autocorrelation_n_eff(daily_returns: Sequence[float]) -> float:
    """Conservative HAC-style daily cluster N_eff capped by unique days.

    Negative autocorrelation is not used to inflate evidence.  Up to five
    positive lags reduce the observed UTC-day count.
    """
    n = len(daily_returns)
    if n < 2:
        return float(n)
    mean = statistics.fmean(daily_returns)
    centered = [value - mean for value in daily_returns]
    variance = sum(value * value for value in centered)
    if variance <= 0:
        return 0.0
    penalty = 1.0
    for lag in range(1, min(5, n - 1) + 1):
        covariance = sum(centered[i] * centered[i - lag] for i in range(lag, n))
        rho = covariance / variance
        if rho > 0:
            penalty += 2.0 * rho
    return max(0.0, min(float(n), float(n) / penalty))


def merge_mark_series(
    pair_runs: Sequence[PairRun], arm: str, initial_equity: Decimal
) -> tuple[list[tuple[int, Decimal]], dict[str, Decimal]]:
    events: list[tuple[int, str, Decimal]] = []
    for run in pair_runs:
        events.extend((stamp, run.pair, contribution) for stamp, contribution in run.mark_series[arm])
    events.sort(key=lambda item: (item[0], item[1]))
    latest = {run.pair: D(0) for run in pair_runs}
    combined: list[tuple[int, Decimal]] = []
    index = 0
    while index < len(events):
        stamp = events[index][0]
        while index < len(events) and events[index][0] == stamp:
            _, pair, contribution = events[index]
            latest[pair] = contribution
            index += 1
        combined.append((stamp, initial_equity + sum(latest.values(), D(0))))
    if not combined:
        raise ChallengerError("combined mark series is empty")
    day_last: dict[str, Decimal] = {}
    for stamp, equity in combined:
        day_last[utc_day(stamp)] = equity
    return combined, day_last


def running_peak_drawdown(series: Sequence[tuple[int, Decimal]]) -> tuple[Decimal, int | None]:
    if not series:
        return D(0), None
    peak = series[0][1]
    worst = D(0)
    worst_stamp: int | None = None
    for stamp, equity in series:
        peak = max(peak, equity)
        if peak <= 0:
            drawdown = D("-1")
        else:
            drawdown = equity / peak - D(1)
        if drawdown < worst:
            worst = drawdown
            worst_stamp = stamp
    return worst, worst_stamp


def monthly_multiples(
    series: Sequence[tuple[int, Decimal]],
    period_start_equity: Decimal,
) -> dict[str, str]:
    month_end: dict[str, Decimal] = {}
    for stamp, equity in series:
        month_end[utc_month(stamp)] = equity
    result: dict[str, str] = {}
    previous = period_start_equity
    for month in sorted(month_end):
        end = month_end[month]
        multiple = end / previous if previous > 0 else D(0)
        result[month] = decimal_text(multiple)
        previous = end
    return result


def pair_monthly_multiples(
    run: PairRun, arm: str, initial_sleeve_equity: Decimal
) -> dict[str, str]:
    series = [
        (stamp, initial_sleeve_equity + contribution)
        for stamp, contribution in run.mark_series[arm]
    ]
    return monthly_multiples(series, initial_sleeve_equity)


def fixed_horizon_entry_quality(
    bars: Sequence[Bar],
    all_signals: Sequence[Signal],
    start_ns: int,
    end_ns: int,
    max_age: int,
) -> dict[str, Any]:
    signed_pips: list[Decimal] = []
    pip = D(INSTRUMENTS[bars[0].pair]["pip"])
    for signal in all_signals:
        if not (start_ns <= signal.decision_ns < end_ns):
            continue
        if signal.fill_bar_ordinal is None:
            continue
        exit_index = signal.fill_bar_ordinal + max_age
        if exit_index >= len(bars) or bars[exit_index].start_ns >= end_ns:
            continue
        entry = bars[signal.fill_bar_ordinal].mid_o
        exit_price = bars[exit_index].mid_o
        signed_pips.append(D(signal.direction) * (exit_price - entry) / pip)
    return {
        "observations": len(signed_pips),
        "direction_accuracy": decimal_text(
            D(sum(value > 0 for value in signed_pips)) / D(len(signed_pips))
            if signed_pips
            else D(0)
        ),
        "mean_signed_pips": decimal_text(safe_mean(signed_pips) or D(0), PIP_QUANTUM),
    }


def exposure_diagnostics(
    trades: Sequence[Trade],
    notional_jpy: Decimal,
) -> dict[str, Any]:
    events: list[tuple[int, int, str, int, str]] = []
    for trade in trades:
        base, quote = trade.pair.split("_")
        events.append((trade.entry_ns, 1, base, trade.direction, trade.signal_id))
        events.append((trade.entry_ns, 1, quote, -trade.direction, trade.signal_id))
        events.append((trade.exit_ns, 0, base, -trade.direction, trade.signal_id))
        events.append((trade.exit_ns, 0, quote, trade.direction, trade.signal_id))
    events.sort(key=lambda row: (row[0], row[1], row[4], row[2]))
    exposure: dict[str, Decimal] = defaultdict(Decimal)
    open_lots = 0
    max_open_lots = 0
    max_gross = D(0)
    max_abs_currency = D(0)
    # Two currency-node rows describe each lot.  Count lot lifecycle separately.
    lot_events: list[tuple[int, int, str]] = []
    for trade in trades:
        lot_events.append((trade.exit_ns, 0, trade.signal_id))
        lot_events.append((trade.entry_ns, 1, trade.signal_id))
    lot_events.sort(key=lambda row: (row[0], row[1], row[2]))
    for _, kind, _ in lot_events:
        open_lots += 1 if kind == 1 else -1
        max_open_lots = max(max_open_lots, open_lots)
        max_gross = max(max_gross, D(open_lots) * notional_jpy)
        if open_lots < 0:
            raise ChallengerError("negative open-lot exposure state")
    for _, kind, currency, signed_direction, _ in events:
        exposure[currency] += D(signed_direction) * notional_jpy
        max_abs_currency = max(max_abs_currency, abs(exposure[currency]))
    if any(value != 0 for value in exposure.values()) or open_lots != 0:
        raise ChallengerError("terminal exposure is nonzero")
    return {
        "maximum_open_lots": max_open_lots,
        "maximum_gross_notional_jpy": decimal_text(max_gross, JPY_QUANTUM),
        "maximum_absolute_currency_exposure_jpy": decimal_text(
            max_abs_currency, JPY_QUANTUM
        ),
        "terminal_signed_currency_exposure_jpy": {
            currency: decimal_text(value, JPY_QUANTUM)
            for currency, value in sorted(exposure.items())
        },
    }


def summarize_configuration(
    pair_runs: Sequence[PairRun],
    corpus: Mapping[str, Sequence[Bar]],
    signals: Mapping[str, Sequence[Signal]],
    period_name: str,
    period_start_ns: int,
    period_end_ns: int,
    config_id: str,
    initial_equity: Decimal,
    notional_jpy: Decimal,
    gross_cap_jpy: Decimal,
    currency_cap_jpy: Decimal,
) -> dict[str, Any]:
    _, max_age = config_parts(config_id)
    all_trades = [trade for run in pair_runs for trade in run.trades]
    total_signals = sum(run.signal_count for run in pair_runs)
    unique_days = sorted({utc_day(signal.decision_ns) for pair in signals.values() for signal in pair if period_start_ns <= signal.decision_ns < period_end_ns})
    elapsed_days = max(1, len(unique_days))
    exposure = exposure_diagnostics(all_trades, notional_jpy)
    if D(exposure["maximum_gross_notional_jpy"]) > gross_cap_jpy:
        raise ChallengerError("portfolio gross cap exceeded")
    if D(exposure["maximum_absolute_currency_exposure_jpy"]) > currency_cap_jpy:
        raise ChallengerError("currency exposure cap exceeded")

    entry_quality = {
        pair: fixed_horizon_entry_quality(
            corpus[pair], signals[pair], period_start_ns, period_end_ns, max_age
        )
        for pair in sorted(corpus)
    }
    arms: dict[str, Any] = {}
    for arm in COST_ARMS:
        pnl_values = [trade.pnl_jpy[arm] for trade in all_trades]
        ratio_values = [trade.return_ratio[arm] for trade in all_trades]
        combined, day_last = merge_mark_series(pair_runs, arm, initial_equity)
        worst_dd, worst_stamp = running_peak_drawdown(combined)
        daily_equities = [day_last[day] for day in sorted(day_last)]
        daily_returns: list[float] = []
        previous = initial_equity
        for equity in daily_equities:
            daily_returns.append(float(equity / previous - D(1)) if previous > 0 else -1.0)
            previous = equity
        n_eff = positive_autocorrelation_n_eff(daily_returns)
        final_equity = combined[-1][1]
        arms[arm] = {
            "completed_round_trips": len(all_trades),
            "total_pnl_jpy": decimal_text(sum(pnl_values, D(0)), JPY_QUANTUM),
            "mean_net_expectancy_jpy": decimal_text(safe_mean(pnl_values) or D(0), JPY_QUANTUM),
            "mean_net_return_ratio": decimal_text(safe_mean(ratio_values) or D(0)),
            "ending_equity_jpy": decimal_text(final_equity, JPY_QUANTUM),
            "ending_equity_multiple": decimal_text(final_equity / initial_equity),
            "monthly_multiples": monthly_multiples(combined, initial_equity),
            "marked_running_peak_max_drawdown": decimal_text(worst_dd),
            "max_drawdown_timestamp": format_epoch_ns(worst_stamp) if worst_stamp is not None else None,
            "currency_time_daily_cluster_count": len(day_last),
            "currency_time_clustered_n_eff": f"{n_eff:.6f}",
            "terminal_mtm_jpy": "0.000000",
            "terminal_inventory": 0,
        }
    for arm in ("EXECUTABLE_BASE", "ADVERSE_STRESS"):
        arms[arm]["cost_drag_vs_raw_jpy"] = decimal_text(
            D(arms["RAW_SIGNAL"]["total_pnl_jpy"]) - D(arms[arm]["total_pnl_jpy"]),
            JPY_QUANTUM,
        )

    pair_metrics: dict[str, Any] = {}
    initial_sleeve = initial_equity / D(len(pair_runs))
    for run in sorted(pair_runs, key=lambda item: item.pair):
        pair_trades = run.trades
        ages = [D(trade.age_completed_bars) for trade in pair_trades]
        wall_ages = [D(trade.wall_age_seconds) for trade in pair_trades]
        pair_metrics[run.pair] = {
            "signals": run.signal_count,
            "fill_eligible": run.fill_eligible_count,
            "filled": run.filled_count,
            "cap_rejected": run.cap_rejected_count,
            "no_future_fill": run.no_future_fill_count,
            "raw_signals_per_observed_utc_day": decimal_text(
                D(run.signal_count) / D(elapsed_days), PIP_QUANTUM
            ),
            "fixed_horizon_entry_quality": entry_quality[run.pair],
            "gross_mfe_mean_pips": decimal_text(
                safe_mean([trade.mfe_pips for trade in pair_trades]) or D(0), PIP_QUANTUM
            ),
            "gross_mae_mean_pips": decimal_text(
                safe_mean([trade.mae_pips for trade in pair_trades]) or D(0), PIP_QUANTUM
            ),
            "inventory_age_completed_bars": {
                "q50": decimal_text(percentile(ages, D("0.50")) or D(0), PIP_QUANTUM),
                "q90": decimal_text(percentile(ages, D("0.90")) or D(0), PIP_QUANTUM),
                "q99": decimal_text(percentile(ages, D("0.99")) or D(0), PIP_QUANTUM),
                "maximum": decimal_text(max(ages, default=D(0)), PIP_QUANTUM),
            },
            "inventory_age_wall_seconds": {
                "q50": decimal_text(percentile(wall_ages, D("0.50")) or D(0), PIP_QUANTUM),
                "q90": decimal_text(percentile(wall_ages, D("0.90")) or D(0), PIP_QUANTUM),
                "q99": decimal_text(percentile(wall_ages, D("0.99")) or D(0), PIP_QUANTUM),
                "maximum": decimal_text(max(wall_ages, default=D(0)), PIP_QUANTUM),
            },
            "exit_reason_counts": dict(
                sorted(
                    {
                        reason: sum(trade.exit_reason == reason for trade in pair_trades)
                        for reason in {trade.exit_reason for trade in pair_trades}
                    }.items()
                )
            ),
            "max_open_lots": run.max_open_lots,
            "terminal_liquidation_count": run.terminal_liquidation_count,
            "terminal_inventory": run.terminal_open_inventory,
            "disposition_root_sha256": run.disposition_root_sha256,
            "filled_signal_id_set_sha256_all_cost_arms": {
                arm: run.filled_signal_id_set_sha256 for arm in COST_ARMS
            },
            "action_root_sha256_all_cost_arms": {
                arm: run.action_root_sha256 for arm in COST_ARMS
            },
            "arm_metrics": {
                arm: {
                    "total_pnl_jpy": decimal_text(
                        sum((trade.pnl_jpy[arm] for trade in pair_trades), D(0)),
                        JPY_QUANTUM,
                    ),
                    "mean_net_expectancy_jpy": decimal_text(
                        safe_mean([trade.pnl_jpy[arm] for trade in pair_trades]) or D(0),
                        JPY_QUANTUM,
                    ),
                    "monthly_sleeve_multiples": pair_monthly_multiples(
                        run, arm, initial_sleeve
                    ),
                }
                for arm in COST_ARMS
            },
        }

    ages_all = [D(trade.age_completed_bars) for trade in all_trades]
    result = {
        "period": period_name,
        "start_utc": format_epoch_ns(period_start_ns),
        "end_utc_exclusive": format_epoch_ns(period_end_ns),
        "config_id": config_id,
        "generated_raw_signals": total_signals,
        "filled_fixed_lots": len(all_trades),
        "cap_rejected": sum(run.cap_rejected_count for run in pair_runs),
        "turnover_jpy": decimal_text(D(2 * len(all_trades)) * notional_jpy, JPY_QUANTUM),
        "turnover_multiple_initial_equity": decimal_text(
            D(2 * len(all_trades)) * notional_jpy / initial_equity
        ),
        "inventory_age_completed_bars": {
            "q50": decimal_text(percentile(ages_all, D("0.50")) or D(0), PIP_QUANTUM),
            "q90": decimal_text(percentile(ages_all, D("0.90")) or D(0), PIP_QUANTUM),
            "q99": decimal_text(percentile(ages_all, D("0.99")) or D(0), PIP_QUANTUM),
        },
        "exposure": exposure,
        "gross_cap_jpy": decimal_text(gross_cap_jpy, JPY_QUANTUM),
        "currency_cap_jpy": decimal_text(currency_cap_jpy, JPY_QUANTUM),
        "hard_guard_violation_count": 0,
        "terminal_inventory": 0,
        "terminal_mtm_jpy": "0.000000",
        "external_orders": 0,
        "arms": arms,
        "pairs": pair_metrics,
    }
    result["summary_sha256"] = embedded_sha256(result, "summary_sha256")
    return result


def parse_contract_timestamp(value: str) -> int:
    if value.endswith("Z") and "." not in value:
        value = value[:-1] + ".000000000Z"
    return parse_epoch_ns(value)


def prereg_periods(prereg: Mapping[str, Any]) -> dict[str, tuple[int, int]]:
    periods = prereg.get("periods")
    if not isinstance(periods, dict):
        raise ChallengerError("prereg periods are missing")
    result: dict[str, tuple[int, int]] = {}
    for key in ("tuning", "walk_forward"):
        item = periods.get(key)
        if not isinstance(item, dict):
            raise ChallengerError(f"prereg period is missing: {key}")
        start = parse_contract_timestamp(str(item["start_utc_inclusive"]))
        end = parse_contract_timestamp(str(item["end_utc_exclusive"]))
        if start >= end:
            raise ChallengerError(f"invalid prereg period: {key}")
        result[key] = (start, end)
    if result["tuning"][1] > result["walk_forward"][0]:
        raise ChallengerError("tuning overlaps walk-forward")
    return result


def validate_prereg(prereg: Mapping[str, Any]) -> None:
    if prereg.get("artifact_id") != CANDIDATE_ID or prereg.get("candidate_id") != CANDIDATE_ID:
        raise ChallengerError("wrong candidate preregistration")
    if prereg.get("status") != "PREREGISTERED_UNEXECUTED":
        raise ChallengerError("preregistration is not frozen-unexecuted")
    authority = prereg.get("authority")
    expected_authority = {
        "paper_only": True,
        "live_authority": False,
        "broker_account_access": False,
        "credential_access": False,
        "order_endpoint": False,
        "external_orders": 0,
        "deploy": False,
        "external_config_mutation": False,
        "network_access": False,
        "feed_or_launchd_mutation": False,
    }
    if authority != expected_authority:
        raise ChallengerError("prereg authority is not exact paper-only NONE")
    if prereg.get("configuration_family", {}).get("configuration_ids") != list(CONFIG_IDS):
        raise ChallengerError("configuration budget is not the exact frozen 12")
    if prereg.get("signal_contract", {}).get("cost_used_for_signal_or_entry_gate") is not False:
        raise ChallengerError("cost gating must be false")
    if prereg.get("position_contract", {}).get("individual_price_stop_loss") is not False:
        raise ChallengerError("individual price SL must be false")
    holdout = prereg.get("periods", {}).get("holdout", {})
    if holdout.get("state") != "UNOPENED" or holdout.get("may_read") is not False:
        raise ChallengerError("holdout must remain UNOPENED and unreadable")
    if prereg.get("selection_contract", {}).get("two_x_or_three_x_used_for_selection") is not False:
        raise ChallengerError("2x/3x leaked into discovery selection")
    prereg_periods(prereg)


def build_source_manifest(
    prereg_sha256: str,
    source_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    schema_hashes = {str(row["schema_sha256"]) for row in source_rows}
    if len(schema_hashes) != 1:
        raise ChallengerError("included pairs do not share one exact source schema")
    manifest = {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "prereg_sha256": prereg_sha256,
        "provider": "OANDA_HISTORICAL_SNAPSHOT",
        "bar_granularity": "M5",
        "price_component": "BID_ASK",
        "completed_only": True,
        "historical_arrival_sequence_heartbeat_available": False,
        "feed_quality_claim_allowed": False,
        "pair_count": len(source_rows),
        "total_completed_bars": sum(int(row["row_count"]) for row in source_rows),
        "schema_sha256": next(iter(schema_hashes)),
        "sources": list(source_rows),
        "integrity_error_count": 0,
    }
    manifest["source_manifest_sha256"] = embedded_sha256(
        manifest, "source_manifest_sha256"
    )
    return manifest


def signal_ledger_bytes(
    signals: Mapping[str, Sequence[Signal]],
    periods: Mapping[str, tuple[int, int]],
    prereg_sha256: str,
    source_manifest_sha256: str,
) -> tuple[bytes, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous = ZERO_SHA256
    period_counts: dict[str, int] = defaultdict(int)
    direction_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"LONG": 0, "SHORT": 0})
    for pair in sorted(signals):
        for signal in signals[pair]:
            period = None
            for name, (start, end) in periods.items():
                if start <= signal.decision_ns < end:
                    period = name.upper()
                    break
            if period is None:
                continue
            row = {
                "schema_version": 1,
                "candidate_id": CANDIDATE_ID,
                "prereg_sha256": prereg_sha256,
                "source_manifest_sha256": source_manifest_sha256,
                "signal_id": signal.signal_id,
                "pair": signal.pair,
                "period": period,
                "direction": "LONG" if signal.direction > 0 else "SHORT",
                "direction_int": signal.direction,
                "decision_bar_ordinal": signal.decision_bar_ordinal,
                "decision_source_time": signal.decision_source_time,
                "decision_timestamp": format_epoch_ns(signal.decision_ns),
                "decision_epoch_ns": signal.decision_ns,
                "fill_bar_ordinal": signal.fill_bar_ordinal,
                "fill_source_time": signal.fill_source_time,
                "fixed_notional_jpy": 1000,
                "ema3": decimal_text(signal.ema3),
                "ema12": decimal_text(signal.ema12),
                "source_row_sha256": signal.source_row_sha256,
                "cost_used_for_signal_or_entry_gate": False,
                "same_signal_id_for_cost_arms": list(COST_ARMS),
                "previous_row_hash": previous,
            }
            row_hash = sha256_bytes(canonical_json_bytes(row))
            row["row_hash"] = row_hash
            previous = row_hash
            rows.append(row)
            period_counts[period] += 1
            direction_counts[pair][row["direction"]] += 1
    raw = b"".join(canonical_json_bytes(row) for row in rows)
    return raw, {
        "row_count": len(rows),
        "terminal_row_hash": previous,
        "signal_id_set_sha256": signal_id_set_sha256(row["signal_id"] for row in rows),
        "period_counts": dict(sorted(period_counts.items())),
        "pair_direction_counts": {
            pair: dict(counts) for pair, counts in sorted(direction_counts.items())
        },
    }


def selection_receipt(tuning_results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for config_id in CONFIG_IDS:
        summary = tuning_results[config_id]
        base_multiple = D(summary["arms"]["EXECUTABLE_BASE"]["ending_equity_multiple"])
        adverse_multiple = D(summary["arms"]["ADVERSE_STRESS"]["ending_equity_multiple"])
        base_expectancy = D(summary["arms"]["EXECUTABLE_BASE"]["mean_net_expectancy_jpy"])
        adverse_expectancy = D(summary["arms"]["ADVERSE_STRESS"]["mean_net_expectancy_jpy"])
        if base_multiple <= 0 or adverse_multiple <= 0:
            minimum_log_growth = float("-inf")
        else:
            minimum_log_growth = min(math.log(float(base_multiple)), math.log(float(adverse_multiple)))
        positive_both = base_expectancy > 0 and adverse_expectancy > 0
        candidates.append(
            {
                "config_id": config_id,
                "positive_base_and_adverse_expectancy": positive_both,
                "base_ending_multiple": decimal_text(base_multiple),
                "adverse_ending_multiple": decimal_text(adverse_multiple),
                "minimum_base_adverse_log_growth": (
                    f"{minimum_log_growth:.12f}" if math.isfinite(minimum_log_growth) else "-Infinity"
                ),
                "summary_sha256": summary["summary_sha256"],
            }
        )
    qualifying = [item for item in candidates if item["positive_base_and_adverse_expectancy"]]
    pool = qualifying or candidates
    selected = sorted(
        pool,
        key=lambda item: (
            -float(item["minimum_base_adverse_log_growth"]),
            item["config_id"],
        ),
    )[0]
    receipt = {
        "schema_version": 1,
        "selection_data": "TUNING_ONLY",
        "objective": "MAXIMIZE_MINIMUM_BASE_ADVERSE_LOG_GROWTH",
        "positive_expectancy_filter_applied": bool(qualifying),
        "diagnostic_fallback_used": not bool(qualifying),
        "walk_forward_metrics_read_before_selection": False,
        "holdout_read": False,
        "two_x_or_three_x_used": False,
        "post_evaluation_leverage_change": False,
        "selected_config_id": selected["config_id"],
        "candidates": candidates,
    }
    receipt["selection_receipt_sha256"] = embedded_sha256(
        receipt, "selection_receipt_sha256"
    )
    return receipt


def selected_final_measurement(walk_summary: Mapping[str, Any]) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    for arm in ("EXECUTABLE_BASE", "ADVERSE_STRESS"):
        months = walk_summary["arms"][arm]["monthly_multiples"]
        arms[arm] = {
            "monthly_multiples": months,
            "months_at_or_above_2x": sum(D(value) >= D(2) for value in months.values()),
            "months_at_or_above_3x": sum(D(value) >= D(3) for value in months.values()),
            "all_full_months_at_or_above_2x": bool(months)
            and all(D(value) >= D(2) for value in months.values()),
        }
    return {
        "classification": "ASPIRATIONAL_FINAL_MEASUREMENT_NOT_SELECTION_INPUT",
        "selection_hash_unchanged_by_threshold": True,
        "arms": arms,
    }


def build_result(
    prereg_sha256: str,
    code_sha256: str,
    source_manifest_file_sha256: str,
    source_manifest_embedded_sha256: str,
    signal_ledger_sha256: str,
    signal_ledger_info: Mapping[str, Any],
    target_freeze: Mapping[str, Any],
    tuning_results: Mapping[str, Mapping[str, Any]],
    walk_results: Mapping[str, Mapping[str, Any]],
    selection: Mapping[str, Any],
    operation_counters: Mapping[str, int],
    parent_baseline: Mapping[str, Any],
) -> dict[str, Any]:
    selected_id = str(selection["selected_config_id"])
    selected_walk = walk_results[selected_id]
    base = selected_walk["arms"]["EXECUTABLE_BASE"]
    adverse = selected_walk["arms"]["ADVERSE_STRESS"]
    month_stable = all(
        D(value) > 1
        for arm in (base, adverse)
        for value in arm["monthly_multiples"].values()
    )
    development_positive = (
        D(base["mean_net_expectancy_jpy"]) > 0
        and D(adverse["mean_net_expectancy_jpy"]) > 0
        and month_stable
        and float(base["currency_time_clustered_n_eff"]) >= 30
        and float(adverse["currency_time_clustered_n_eff"]) >= 30
        and selected_walk["terminal_inventory"] == 0
        and selected_walk["hard_guard_violation_count"] == 0
    )
    result = {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "classification": "UNADMITTED_CHALLENGER",
        "official_historical_replay_ordinal": 1,
        "prereg_sha256": prereg_sha256,
        "runner_sha256": code_sha256,
        "source_manifest_file_sha256": source_manifest_file_sha256,
        "source_manifest_embedded_sha256": source_manifest_embedded_sha256,
        "raw_signal_ledger_sha256": signal_ledger_sha256,
        "raw_signal_ledger": dict(signal_ledger_info),
        "target_freeze": dict(target_freeze),
        "selection_receipt": dict(selection),
        "tuning_results": dict(tuning_results),
        "walk_forward_results": dict(walk_results),
        "selected_config_id": selected_id,
        "selected_walk_forward_summary_sha256": selected_walk["summary_sha256"],
        "development_positive_diagnostic": development_positive,
        "development_positive_is_admission": False,
        "admission_status": "UNADMITTED_CHALLENGER",
        "profit_unproven": True,
        "holdout": {
            "label": "FUTURE_FX_HOLDOUT_AFTER_2026_07_15",
            "state": "UNOPENED",
            "read_count": 0,
        },
        "aspirational_2x_3x_final_measurement": selected_final_measurement(selected_walk),
        "operation_counters": dict(operation_counters),
        "parent_baseline": dict(parent_baseline),
        "r5_unchanged_claim_requires_post_run_hash_readback": True,
        "active_shadow_mutated": False,
        "feed_connected": False,
        "forward_observation_started": False,
        "llm_calls": 0,
        "external_orders": 0,
        "authority": {
            "paper_only": True,
            "live_authority": False,
            "broker_account_access": False,
            "credential_access": False,
            "order_endpoint": False,
            "deploy": False,
            "external_config_mutation": False,
        },
        "admission_blockers": [
            "UNTOUCHED_FUTURE_HOLDOUT_UNOPENED",
            "ACCOUNTING_ORACLE_V2_REMAINS_NON_ADMISSIBLE_FOR_CAUSAL_SIGNAL_CLAIMS",
            "TWO_MONTH_OPENED_WALK_FORWARD_IS_DEVELOPMENT_ONLY",
            "NO_ACTIVE_SHADOW_INTEGRATION_OR_FORWARD_RECEIPT",
        ],
    }
    result["result_sha256"] = embedded_sha256(result, "result_sha256")
    return result


def challenger_packet(
    prereg: Mapping[str, Any],
    prereg_sha256: str,
    code_sha256: str,
    source_manifest_file_sha256: str,
    source_manifest_embedded_sha256: str,
    result_file_sha256: str,
    result: Mapping[str, Any],
    signal_ledger_sha256: str,
) -> dict[str, Any]:
    selected = str(result["selected_config_id"])
    selection = result["selection_receipt"]
    packet = {
        "schema_version": 1,
        "packet_type": "UNADMITTED_CHALLENGER",
        "candidate_id": CANDIDATE_ID,
        "status": "UNADMITTED_CHALLENGER",
        "prereg_sha256": prereg_sha256,
        "source_manifest_file_sha256": source_manifest_file_sha256,
        "source_manifest_embedded_sha256": source_manifest_embedded_sha256,
        "raw_signal_ledger_sha256": signal_ledger_sha256,
        "result_file_sha256": result_file_sha256,
        "result_embedded_sha256": result["result_sha256"],
        "runner_sha256": code_sha256,
        "selection_receipt_sha256": selection["selection_receipt_sha256"],
        "target_freeze_sha256": result["target_freeze"]["target_freeze_sha256"],
        "exact_formula": prereg["signal_contract"],
        "symbols": prereg["universe"],
        "warmup": {
            "completed_m5_bars": 12,
            "ema3_seed": "arithmetic mean first 3 completed mid closes",
            "ema12_seed": "arithmetic mean first 12 completed mid closes",
        },
        "decision_fill_chronology": prereg["chronology"],
        "size_and_caps": prereg["position_contract"],
        "selected_exit_policy": {
            "config_id": selected,
            "policy": config_parts(selected)[0],
            "finite_max_age_completed_m5_bars": config_parts(selected)[1],
            "selection_data": "TUNING_ONLY",
            "walk_forward_used_for_selection": False,
        },
        "natural_proposal_counts": result["raw_signal_ledger"]["period_counts"],
        "same_signal_id_cost_arms": list(COST_ARMS),
        "cost_gate_at_entry": False,
        "profit_unproven": True,
        "holdout_unopened": True,
        "holdout_read_count": 0,
        "shadow_interface": {
            "content_addressed_proposal_stream_available": True,
            "active_shadow_mutated": False,
            "auto_enable_allowed": False,
            "requires_separate_forward_shadow_registration": True,
        },
        "external_orders": 0,
        "authority": prereg["authority"],
    }
    packet["packet_sha256"] = embedded_sha256(packet, "packet_sha256")
    return packet


def official_build(prereg_path: Path, input_root: Path, output_root: Path) -> dict[str, Any]:
    if output_root.exists() or output_root.is_symlink():
        raise ChallengerError(f"official output already exists; no rerun/overwrite: {output_root}")
    prereg_raw = read_regular_no_follow(prereg_path)
    prereg = strict_json_loads(prereg_raw.decode("utf-8"))
    if not isinstance(prereg, dict):
        raise ChallengerError("preregistration must be an object")
    validate_prereg(prereg)
    prereg_sha = sha256_bytes(prereg_raw)
    code_path = Path(__file__).resolve()
    code_sha = sha256_bytes(read_regular_no_follow(code_path))
    periods = prereg_periods(prereg)
    source_contract = prereg["source_contract"]
    declared_files = source_contract["files"]
    expected_pairs = ["EUR_USD", "USD_JPY", "AUD_USD"]
    if sorted(declared_files) != sorted(expected_pairs):
        raise ChallengerError("source pair set differs from frozen universe")

    corpus: dict[str, list[Bar]] = {}
    source_rows: list[dict[str, Any]] = []
    for pair in expected_pairs:
        descriptor = declared_files[pair]
        path = input_root / str(descriptor["relative_path"])
        bars, source_row = load_bars(path, pair, str(descriptor["sha256"]))
        corpus[pair] = bars
        source_rows.append(source_row)
    manifest = build_source_manifest(prereg_sha, source_rows)
    manifest_raw = canonical_json_bytes(manifest)
    source_manifest_file_sha = sha256_bytes(manifest_raw)

    signals = {pair: ema_signals(bars) for pair, bars in corpus.items()}
    target_freeze = freeze_mfe_targets(
        corpus,
        signals,
        periods["tuning"][0],
        periods["tuning"][1],
    )
    ledger_raw, ledger_info = signal_ledger_bytes(
        signals,
        periods,
        prereg_sha,
        manifest["source_manifest_sha256"],
    )
    ledger_sha = sha256_bytes(ledger_raw)

    position_contract = prereg["position_contract"]
    notional = D(position_contract["fixed_notional_jpy_per_lot"])
    initial_equity = D(position_contract["initial_equity_jpy"])
    pair_cap = int(position_contract["pair_open_lot_cap"])
    gross_cap = D(position_contract["gross_notional_cap_jpy"])
    currency_cap = D(position_contract["currency_absolute_exposure_cap_jpy"])

    tuning_results: dict[str, Any] = {}
    for config_id in CONFIG_IDS:
        runs = [
            simulate_pair(
                corpus[pair],
                signals[pair],
                periods["tuning"][0],
                periods["tuning"][1],
                config_id,
                target_freeze,
                notional,
                pair_cap,
            )
            for pair in expected_pairs
        ]
        tuning_results[config_id] = summarize_configuration(
            runs,
            corpus,
            signals,
            "TUNING",
            periods["tuning"][0],
            periods["tuning"][1],
            config_id,
            initial_equity,
            notional,
            gross_cap,
            currency_cap,
        )
    selection = selection_receipt(tuning_results)

    walk_results: dict[str, Any] = {}
    for config_id in CONFIG_IDS:
        runs = [
            simulate_pair(
                corpus[pair],
                signals[pair],
                periods["walk_forward"][0],
                periods["walk_forward"][1],
                config_id,
                target_freeze,
                notional,
                pair_cap,
            )
            for pair in expected_pairs
        ]
        walk_results[config_id] = summarize_configuration(
            runs,
            corpus,
            signals,
            "WALK_FORWARD",
            periods["walk_forward"][0],
            periods["walk_forward"][1],
            config_id,
            initial_equity,
            notional,
            gross_cap,
            currency_cap,
        )

    operation_counters = {
        "source_files_decoded": len(corpus),
        "completed_bars_decoded": sum(len(bars) for bars in corpus.values()),
        "ema_signal_streams_computed": len(corpus),
        "raw_signals_content_addressed": int(ledger_info["row_count"]),
        "post_entry_configurations_evaluated_tuning": len(CONFIG_IDS),
        "post_entry_configurations_evaluated_walk_forward": len(CONFIG_IDS),
        "cost_arms_fanned_out": len(COST_ARMS),
        "llm_calls": 0,
        "external_orders": 0,
    }
    result = build_result(
        prereg_sha,
        code_sha,
        source_manifest_file_sha,
        manifest["source_manifest_sha256"],
        ledger_sha,
        ledger_info,
        target_freeze,
        tuning_results,
        walk_results,
        selection,
        operation_counters,
        prereg["parent_baseline"],
    )
    result_raw = canonical_json_bytes(result)
    result_file_sha = sha256_bytes(result_raw)
    packet = challenger_packet(
        prereg,
        prereg_sha,
        code_sha,
        source_manifest_file_sha,
        manifest["source_manifest_sha256"],
        result_file_sha,
        result,
        ledger_sha,
    )
    packet_name = f"UNADMITTED_CHALLENGER_PACKET_{packet['packet_sha256']}.json"
    packet_raw = canonical_json_bytes(packet)

    output_root.parent.mkdir(parents=True, exist_ok=True)
    os.mkdir(output_root, 0o700)
    directory_fd = os.open(output_root.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    atomic_write(output_root / SOURCE_MANIFEST_NAME, manifest_raw)
    atomic_write(output_root / SIGNAL_LEDGER_NAME, ledger_raw)
    atomic_write(output_root / RESULT_NAME, result_raw)
    atomic_write(output_root / packet_name, packet_raw)
    artifacts = {}
    for name in (SOURCE_MANIFEST_NAME, SIGNAL_LEDGER_NAME, RESULT_NAME, packet_name):
        raw = read_regular_no_follow(output_root / name)
        artifacts[name] = {"sha256": sha256_bytes(raw), "size_bytes": len(raw)}
    artifact_manifest = {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "classification": "UNADMITTED_CHALLENGER",
        "prereg_sha256": prereg_sha,
        "runner_sha256": code_sha,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "packet_path": packet_name,
        "packet_embedded_sha256": packet["packet_sha256"],
        "profit_unproven": True,
        "holdout_unopened": True,
        "external_orders": 0,
        "complete": True,
    }
    artifact_manifest["artifact_manifest_sha256"] = embedded_sha256(
        artifact_manifest, "artifact_manifest_sha256"
    )
    atomic_write(output_root / ARTIFACT_MANIFEST_NAME, canonical_json_bytes(artifact_manifest))
    verified = verify_artifacts(prereg_path, input_root, output_root)
    return {
        "candidate_id": CANDIDATE_ID,
        "status": "UNADMITTED_CHALLENGER",
        "selected_config_id": result["selected_config_id"],
        "tuning_raw_signals": ledger_info["period_counts"].get("TUNING", 0),
        "walk_forward_raw_signals": ledger_info["period_counts"].get("WALK_FORWARD", 0),
        "packet_sha256": packet["packet_sha256"],
        "result_sha256": result["result_sha256"],
        "artifact_manifest_sha256": artifact_manifest["artifact_manifest_sha256"],
        "verified": verified["verified"],
        "profit_unproven": True,
        "holdout_unopened": True,
        "external_orders": 0,
    }


def verify_artifacts(prereg_path: Path, input_root: Path, output_root: Path) -> dict[str, Any]:
    prereg_raw = read_regular_no_follow(prereg_path)
    prereg = strict_json_loads(prereg_raw.decode("utf-8"))
    if not isinstance(prereg, dict):
        raise ChallengerError("prereg must be an object")
    validate_prereg(prereg)
    prereg_sha = sha256_bytes(prereg_raw)
    manifest_path = output_root / ARTIFACT_MANIFEST_NAME
    artifact_manifest = strict_json_loads(read_regular_no_follow(manifest_path).decode("utf-8"))
    if not isinstance(artifact_manifest, dict):
        raise ChallengerError("artifact manifest must be an object")
    if artifact_manifest.get("artifact_manifest_sha256") != embedded_sha256(
        artifact_manifest, "artifact_manifest_sha256"
    ):
        raise ChallengerError("artifact manifest embedded SHA mismatch")
    if artifact_manifest.get("prereg_sha256") != prereg_sha:
        raise ChallengerError("artifact manifest prereg SHA mismatch")
    artifacts = artifact_manifest.get("artifacts")
    if not isinstance(artifacts, dict) or artifact_manifest.get("artifact_count") != len(artifacts):
        raise ChallengerError("artifact inventory mismatch")
    expected_names = {
        SOURCE_MANIFEST_NAME,
        SIGNAL_LEDGER_NAME,
        RESULT_NAME,
        str(artifact_manifest.get("packet_path")),
    }
    actual_names = {
        path.name
        for path in output_root.iterdir()
        if path.name != ARTIFACT_MANIFEST_NAME
    }
    if set(artifacts) != expected_names or actual_names != expected_names:
        raise ChallengerError("artifact exact-set mismatch")
    for name, descriptor in artifacts.items():
        raw = read_regular_no_follow(output_root / name)
        if descriptor != {"sha256": sha256_bytes(raw), "size_bytes": len(raw)}:
            raise ChallengerError(f"artifact hash/size mismatch: {name}")

    source_manifest = strict_json_loads(
        read_regular_no_follow(output_root / SOURCE_MANIFEST_NAME).decode("utf-8")
    )
    if not isinstance(source_manifest, dict) or source_manifest.get(
        "source_manifest_sha256"
    ) != embedded_sha256(source_manifest, "source_manifest_sha256"):
        raise ChallengerError("source manifest embedded SHA mismatch")
    for source in source_manifest.get("sources", []):
        pair = str(source["pair"])
        descriptor = prereg["source_contract"]["files"][pair]
        source_path = input_root / str(descriptor["relative_path"])
        if sha256_bytes(read_regular_no_follow(source_path)) != descriptor["sha256"]:
            raise ChallengerError(f"source drift during verification: {pair}")

    result = strict_json_loads(read_regular_no_follow(output_root / RESULT_NAME).decode("utf-8"))
    if not isinstance(result, dict) or result.get("result_sha256") != embedded_sha256(
        result, "result_sha256"
    ):
        raise ChallengerError("result embedded SHA mismatch")
    packet_path = output_root / str(artifact_manifest["packet_path"])
    packet = strict_json_loads(read_regular_no_follow(packet_path).decode("utf-8"))
    if not isinstance(packet, dict) or packet.get("packet_sha256") != embedded_sha256(
        packet, "packet_sha256"
    ):
        raise ChallengerError("packet embedded SHA mismatch")
    if packet_path.name != f"UNADMITTED_CHALLENGER_PACKET_{packet['packet_sha256']}.json":
        raise ChallengerError("packet filename is not content addressed")
    if packet.get("result_file_sha256") != sha256_bytes(
        read_regular_no_follow(output_root / RESULT_NAME)
    ):
        raise ChallengerError("packet result binding mismatch")
    if packet.get("raw_signal_ledger_sha256") != sha256_bytes(
        read_regular_no_follow(output_root / SIGNAL_LEDGER_NAME)
    ):
        raise ChallengerError("packet signal-ledger binding mismatch")
    if (
        packet.get("profit_unproven") is not True
        or packet.get("holdout_unopened") is not True
        or packet.get("external_orders") != 0
        or packet.get("authority") != prereg["authority"]
    ):
        raise ChallengerError("packet safety/admission boundary mismatch")
    if result.get("selected_config_id") != packet.get("selected_exit_policy", {}).get(
        "config_id"
    ):
        raise ChallengerError("selected config binding mismatch")
    return {
        "candidate_id": CANDIDATE_ID,
        "verified": True,
        "packet_sha256": packet["packet_sha256"],
        "result_sha256": result["result_sha256"],
        "artifact_manifest_sha256": artifact_manifest["artifact_manifest_sha256"],
        "profit_unproven": True,
        "holdout_unopened": True,
        "external_orders": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--prereg", type=Path, required=True)
    build.add_argument("--input-root", type=Path, required=True)
    build.add_argument("--output-root", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--prereg", type=Path, required=True)
    verify.add_argument("--input-root", type=Path, required=True)
    verify.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "build":
        result = official_build(args.prereg, args.input_root, args.output_root)
    else:
        result = verify_artifacts(args.prereg, args.input_root, args.output_root)
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ChallengerError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        raise SystemExit(2) from exc
