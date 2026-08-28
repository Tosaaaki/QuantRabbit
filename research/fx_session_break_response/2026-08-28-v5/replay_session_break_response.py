#!/usr/bin/env python3
"""One-shot, offline replay for FX_SESSION_BREAK_RESPONSE_SURFACE_V5.

The module deliberately has no network, credential, broker, order, launchd, or
Git interface.  It verifies and decodes only the two preregistered byte-prefix
phases of the immutable OANDA M5 BID/ASK capture.  Discovery selects and seals
one config before the locked validation suffix is semantically decoded.
"""
from __future__ import annotations

import argparse
import calendar
import datetime as dt
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[2]
PREREG_PATH = ROOT / "PREREGISTRATION.json"
RESULT_PATH = ROOT / "result.json"
PACKET_PATH = ROOT / "evidence_packet.json"
BAR_SECONDS = 300
EPSILON = 1e-12
EXPECTED_PREREG_FILE_SHA256 = "6542095c980ec421d779f21b909dec8f860c5b745a349c8dbafea904d1be4b7e"
EXPECTED_PREREG_CANONICAL_SHA256 = "37981d1cf749d397f4626aa60c379ddf63e69a700d4278647e078bd2872802ac"
LONDON = ZoneInfo("Europe/London")
UTC = dt.timezone.utc
FIELDS = {"o": 0, "h": 1, "l": 2, "c": 3}
ARMS = ("RAW_SIGNAL", "EXECUTABLE_BASE", "ADVERSE_STRESS")


def canonical(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
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


def parse_time(value: str) -> int:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("timestamp must be canonical UTC-Z text")
    body = value[:-1]
    if "." in body:
        head, fraction = body.split(".", 1)
        body = head + "." + fraction[:6]
    parsed = dt.datetime.fromisoformat(body + "+00:00")
    if parsed.tzinfo is None:
        raise ValueError("timestamp must be timezone aware")
    return int(parsed.timestamp())


def iso_utc(seconds: int) -> str:
    return dt.datetime.fromtimestamp(seconds, tz=UTC).strftime(
        "%Y-%m-%dT%H:%M:%S.000000Z"
    )


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(high, max(low, value))


def quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile requires values")
    return float(np.quantile(np.asarray(values, dtype=float), probability,
                             method="linear"))


def pip_size(pair: str) -> float:
    return 0.01 if pair.endswith("_JPY") else 0.0001


def signed_pips(pair: str, side: str, entry: float, exit_: float) -> float:
    direction = 1.0 if side == "LONG" else -1.0
    return direction * (exit_ - entry) / pip_size(pair)


def add_months(value: dt.datetime, months: int) -> dt.datetime:
    base = value.month - 1 + months
    year = value.year + base // 12
    month = base % 12 + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return value.replace(year=year, month=month, day=day)


def anchored_month(seconds: int, split_start: int) -> str:
    start = dt.datetime.fromtimestamp(split_start, tz=UTC)
    value = dt.datetime.fromtimestamp(seconds, tz=UTC)
    index = 0
    while value >= add_months(start, index + 1):
        index += 1
    return f"M{index + 1:02d}"


@dataclass(frozen=True, slots=True)
class Bar:
    time: int
    bid: tuple[float, float, float, float]
    ask: tuple[float, float, float, float]
    volume: int

    def mid(self, field: str) -> float:
        index = FIELDS[field]
        return (self.bid[index] + self.ask[index]) / 2.0


@dataclass(slots=True)
class Observation:
    pair: str
    session: str
    local_day: str
    decision_time: int
    mode: str | None
    break_side: int
    trade_side: str | None
    rail_low: float
    rail_high: float
    rail_log_range: float
    displacement: float
    geometry: float
    path_efficiency: float
    settle: float
    persist_or_reverse: float
    volume_count: int
    activity: float | None = None
    breadth: float | None = None
    common_usd_sign: int = 0
    usd_component: float = 0.0
    ambiguous: bool = False


def load_preregistration() -> dict:
    payload = PREREG_PATH.read_bytes()
    if sha_bytes(payload) != EXPECTED_PREREG_FILE_SHA256:
        raise ValueError("preregistration file SHA-256 mismatch")
    return json.loads(payload)


def validate_preregistration(prereg: dict) -> dict:
    if sha_bytes(canonical(prereg)) != EXPECTED_PREREG_CANONICAL_SHA256:
        raise ValueError("preregistration canonical SHA-256 mismatch")
    configs = prereg["family"]["configs"]
    checks = {
        "candidate": prereg["candidate_id"]
        == "FX_SESSION_BREAK_RESPONSE_SURFACE_V5",
        "family_size": len(configs) == 128
        and len({row["config_id"] for row in configs}) == 128,
        "selection_size": sum(row["selection_eligible"] for row in configs)
        == 32,
        "dataset": prereg["input"]["canonical_dataset_sha256"]
        == "721904751fc1d590a64c7cefd0a533e7df314f043b10783c116d2a82793f14fb",
        "boundary": prereg["input"]["decoder_exclusive_end_utc"]
        == "2025-08-28T04:05:00.000000Z",
        "cost_gate": prereg["indicator"]["cost_is_entry_gate"] is False
        and prereg["execution_arms"]["cost_gate"] is False,
        "authority": prereg["authority"]
        == {
            "offline_only": True,
            "network_attempts_allowed": 0,
            "credential_reads_allowed": 0,
            "broker_mutation_allowed": False,
            "external_orders_allowed": 0,
            "launchd_actions_allowed": 0,
            "git_actions_allowed": 0,
        },
        "resamples": prereg["selection"]["bootstrap_resamples"] == 10000,
        "bootstrap_block": prereg["selection"]["bootstrap_block_days"] == 5,
        "units": prereg["execution_arms"]["units"] == 1000,
        "chronology": prereg["chronology"]["expected_step_seconds"]
        == BAR_SECONDS,
        "epsilon": prereg["indicator"]["epsilon"] == EPSILON,
        "cost_arms": prereg["execution_arms"]["base_slippage_pips_per_side"]
        == 0.3
        and prereg["execution_arms"]["adverse_slippage_pips_per_side"] == 0.9
        and prereg["execution_arms"]["fee_pips_per_side"] == 0.0,
        "latency": prereg["execution_arms"]["latency_sensitivity_bars"] == 1,
        "leverage_cap": prereg["portfolio_and_reporting"][
            "gross_leverage_observation_cap"
        ] == 20.0,
        "family_axes": {
            key: set(value) for key, value in prereg["family"]["dimensions"].items()
        } == {
            "session": {"LONDON_MIDDAY", "LONDON_FIX"},
            "mode": {"ACCEPT_CONTINUATION", "REJECT_FADE"},
            "displacement_quantile": {"Q50", "Q67"},
            "geometry_quantile": {"Q50", "Q67"},
            "breadth": {"ANY", "MODE_MATCHED"},
            "activity": {"ANY", "MODE_MATCHED"},
            "horizon_bars": {24, 48},
        },
    }
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        raise ValueError("preregistration mismatch: " + ",".join(failed))
    return checks


def read_exact_prefix(path: Path, length: int, expected_sha: str) -> bytes:
    remaining = length
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    with path.open("rb", buffering=0) as handle:
        while remaining:
            chunk = handle.read(min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError("source ended before preregistered byte boundary")
            chunks.append(chunk)
            digest.update(chunk)
            remaining -= len(chunk)
    if digest.hexdigest() != expected_sha:
        raise ValueError(f"prefix hash mismatch: {path}")
    return b"".join(chunks)


def read_exact_suffix(
    path: Path,
    prefix_length: int,
    full_length: int,
    prefix_sha: str,
    full_sha: str,
) -> bytes:
    # Both seals are checked before validation bytes are returned to the
    # semantic decoder.  No JSON/price value is decoded in this byte phase.
    prefix = read_exact_prefix(path, prefix_length, prefix_sha)
    del prefix
    full = read_exact_prefix(path, full_length, full_sha)
    return full[prefix_length:]


def parse_bar(raw: bytes, pair: str, start: int, end: int) -> Bar:
    row = json.loads(raw)
    if row.get("schema") != "QR_OANDA_HISTORICAL_M5_BA_ROW_V1":
        raise ValueError("unexpected candle schema")
    if row.get("instrument") != pair or row.get("granularity") != "M5":
        raise ValueError("candle identity mismatch")
    if row.get("price_component") != "BA" or row.get("complete") is not True:
        raise ValueError("candle is not complete BID/ASK")
    timestamp = parse_time(row.get("time_utc"))
    # The timestamp is checked before price or volume fields are touched.
    if not start <= timestamp < end:
        raise AssertionError("row outside authorized semantic decode interval")
    if row.get("volume_semantics") != "OANDA_PRICE_COUNT_NOT_TRADED_VOLUME":
        raise ValueError("volume semantics changed")
    volume = row.get("volume")
    if isinstance(volume, bool) or not isinstance(volume, int) or volume < 0:
        raise ValueError("invalid OANDA price-count volume")
    sides = []
    for side_name in ("bid", "ask"):
        values = row.get(side_name)
        if not isinstance(values, dict) or set(values) != {"o", "h", "l", "c"}:
            raise ValueError("invalid BID/ASK OHLC shape")
        numeric = tuple(float(values[field]) for field in ("o", "h", "l", "c"))
        if not all(math.isfinite(value) and value > 0 for value in numeric):
            raise ValueError("non-positive/nonfinite price")
        if not numeric[2] <= min(numeric[0], numeric[3]) <= max(
            numeric[0], numeric[3]
        ) <= numeric[1]:
            raise ValueError("invalid OHLC geometry")
        sides.append(numeric)
    bid, ask = sides
    if any(bid[index] > ask[index] for index in range(4)):
        raise ValueError("crossed BID/ASK")
    return Bar(timestamp, bid, ask, volume)


def load_discovery_phase(prereg: dict) -> tuple[dict[str, list[Bar]], dict]:
    dataset_root = REPO_ROOT / prereg["input"]["dataset_root"]
    manifest_path = dataset_root / "manifest.json"
    gap_path = dataset_root / "gap_report.json"
    if sha_file(manifest_path) != prereg["input"]["manifest_sha256"]:
        raise ValueError("dataset manifest bytes changed")
    if sha_file(gap_path) != prereg["input"]["gap_report_sha256"]:
        raise ValueError("gap report bytes changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("canonical_dataset_sha256") != prereg["input"][
        "canonical_dataset_sha256"
    ]:
        raise ValueError("canonical dataset seal mismatch")
    if manifest.get("external_order_attempts") != 0 or manifest.get(
        "external_orders"
    ) != 0:
        raise ValueError("capture authority mismatch")
    start = parse_time(prereg["splits"]["calibration"]["from_utc"])
    end = parse_time(prereg["splits"]["discovery"]["to_utc"])
    bars_by_pair = {}
    audit = {
        "canonical_dataset_sha256": manifest["canonical_dataset_sha256"],
        "manifest_sha256": prereg["input"]["manifest_sha256"],
        "winner_locked_before_validation_decode": False,
        "validation_rows_decoded_before_winner_lock": 0,
        "post_boundary_price_or_volume_rows_decoded": 0,
        "post_boundary_label_rows_computed": 0,
        "pairs": {},
    }
    for pair in prereg["input"]["symbols"]:
        contract = prereg["input"]["discovery_prefix_contract"][pair]
        payload = read_exact_prefix(
            dataset_root / contract["path"],
            contract["exclusive_byte_offset"],
            contract["prefix_sha256"],
        )
        lines = payload.splitlines()
        if len(lines) != contract["prefix_rows"]:
            raise ValueError(f"discovery prefix row mismatch: {pair}")
        rows = [parse_bar(line, pair, start, end) for line in lines]
        if any(rows[index + 1].time <= rows[index].time for index in range(len(rows) - 1)):
            raise ValueError(f"non-increasing bars: {pair}")
        bars_by_pair[pair] = rows
        audit["pairs"][pair] = {
            "discovery_prefix_bytes": len(payload),
            "discovery_rows_decoded": len(rows),
            "discovery_prefix_sha256": sha_bytes(payload),
        }
    return bars_by_pair, audit


def load_validation_phase(
    prereg: dict, audit: dict, winner_lock_sha256: str
) -> dict[str, list[Bar]]:
    if not winner_lock_sha256 or audit["validation_rows_decoded_before_winner_lock"] != 0:
        raise AssertionError("validation decoder requires a pre-existing winner lock")
    dataset_root = REPO_ROOT / prereg["input"]["dataset_root"]
    start = parse_time(prereg["splits"]["locked_internal_validation"]["from_utc"])
    end = parse_time(prereg["splits"]["locked_internal_validation"]["to_utc"])
    bars_by_pair = {}
    total = 0
    for pair in prereg["input"]["symbols"]:
        prefix = prereg["input"]["discovery_prefix_contract"][pair]
        full = prereg["input"]["byte_prefix_contract"][pair]
        payload = read_exact_suffix(
            dataset_root / full["path"],
            prefix["exclusive_byte_offset"],
            full["exclusive_byte_offset"],
            prefix["prefix_sha256"],
            full["prefix_sha256"],
        )
        lines = payload.splitlines()
        expected_rows = full["prefix_rows"] - prefix["prefix_rows"]
        if len(lines) != expected_rows:
            raise ValueError(f"validation suffix row mismatch: {pair}")
        rows = [parse_bar(line, pair, start, end) for line in lines]
        if any(rows[index + 1].time <= rows[index].time for index in range(len(rows) - 1)):
            raise ValueError(f"non-increasing validation bars: {pair}")
        bars_by_pair[pair] = rows
        total += len(rows)
        audit["pairs"][pair]["validation_suffix_bytes"] = len(payload)
        audit["pairs"][pair]["validation_rows_decoded_after_winner_lock"] = len(rows)
        audit["pairs"][pair]["validation_suffix_sha256"] = sha_bytes(payload)
    audit["winner_locked_before_validation_decode"] = True
    audit["winner_lock_sha256"] = winner_lock_sha256
    audit["validation_rows_decoded_after_winner_lock"] = total
    return bars_by_pair


def minute_grid(start_minute: int, end_minute: int) -> list[int]:
    return list(range(start_minute, end_minute + 1, 5))


def local_timestamp(day: dt.date, minute: int) -> int:
    hour, minute_part = divmod(minute, 60)
    local = dt.datetime(day.year, day.month, day.day, hour, minute_part,
                        tzinfo=LONDON)
    return int(local.astimezone(UTC).timestamp())


def utc_timestamp(day: dt.date, minute: int) -> int:
    hour, minute_part = divmod(minute, 60)
    return int(dt.datetime(day.year, day.month, day.day, hour, minute_part,
                           tzinfo=UTC).timestamp())


def schedule(session: str, day: dt.date) -> tuple[list[int], list[int]]:
    if session == "LONDON_MIDDAY":
        reference = [utc_timestamp(day, minute) for minute in minute_grid(0, 355)]
        event = [local_timestamp(day, minute) for minute in minute_grid(480, 715)]
    elif session == "LONDON_FIX":
        reference = [local_timestamp(day, minute) for minute in minute_grid(480, 715)]
        event = [local_timestamp(day, minute) for minute in minute_grid(720, 955)]
    else:
        raise ValueError(f"unknown session: {session}")
    return reference, event


def _cube_root(value: float) -> float:
    return max(0.0, value) ** (1.0 / 3.0)


def make_observation(
    pair: str,
    session: str,
    local_day: str,
    reference: list[Bar],
    event: list[Bar],
) -> Observation:
    rail_high = max(bar.mid("h") for bar in reference)
    rail_low = min(bar.mid("l") for bar in reference)
    rail_range = math.log(rail_high / rail_low)
    if not math.isfinite(rail_range) or rail_range <= 0:
        raise ValueError("reference rail has no positive log range")
    event_open = event[0].mid("o")
    event_close = event[-1].mid("c")
    event_high = max(bar.mid("h") for bar in event)
    event_low = min(bar.mid("l") for bar in event)
    returns = []
    prior = event_open
    for bar in event:
        close = bar.mid("c")
        returns.append(math.log(close / prior))
        prior = close
    path_efficiency = abs(sum(returns)) / (sum(abs(value) for value in returns) + EPSILON)
    touched_upper = event_high > rail_high
    touched_lower = event_low < rail_low
    decision_time = event[-1].time + BAR_SECONDS
    volume_count = sum(bar.volume for bar in event)
    usd_q = 1 if pair == "USD_JPY" else -1
    usd_component = usd_q * math.log(event_close / event_open) / (rail_range + EPSILON)
    if touched_upper == touched_lower:
        return Observation(
            pair, session, local_day, decision_time, None, 0, None,
            rail_low, rail_high, rail_range, 0.0, 0.0, path_efficiency,
            0.0, 0.0, volume_count, usd_component=usd_component,
            ambiguous=touched_upper and touched_lower,
        )
    break_side = 1 if touched_upper else -1
    closes = [bar.mid("c") for bar in event]
    last_three_sum = sum(returns[-3:])
    final_inside = rail_low < event_close < rail_high
    if break_side > 0:
        accept_structure = closes[-1] > rail_high and closes[-2] > rail_high
        reject_structure = final_inside and last_three_sum < 0
    else:
        accept_structure = closes[-1] < rail_low and closes[-2] < rail_low
        reject_structure = final_inside and last_three_sum > 0
    if accept_structure:
        displacement = abs(math.log(event_close / event_open)) / (rail_range + EPSILON)
        if break_side > 0:
            settle = clamp((event_close - rail_high) / (event_high - rail_high + EPSILON))
            persist = sum(value > rail_high for value in closes[-6:]) / 6.0
        else:
            settle = clamp((rail_low - event_close) / (rail_low - event_low + EPSILON))
            persist = sum(value < rail_low for value in closes[-6:]) / 6.0
        geometry = _cube_root(path_efficiency * settle * persist)
        mode = "ACCEPT_CONTINUATION"
        trade_side = "LONG" if break_side > 0 else "SHORT"
        component = persist
    elif reject_structure:
        extreme = event_high if break_side > 0 else event_low
        displacement = abs(math.log(extreme / event_open)) / (rail_range + EPSILON)
        if break_side > 0:
            settle = clamp((event_high - event_close) / (event_high - rail_high + EPSILON))
        else:
            settle = clamp((event_close - event_low) / (rail_low - event_low + EPSILON))
        reverse = clamp(
            -break_side * last_three_sum
            / (sum(abs(value) for value in returns[-3:]) + EPSILON)
        )
        geometry = _cube_root((1.0 - path_efficiency) * settle * reverse)
        mode = "REJECT_FADE"
        trade_side = "SHORT" if break_side > 0 else "LONG"
        component = reverse
    else:
        displacement = geometry = settle = component = 0.0
        mode = None
        trade_side = None
    return Observation(
        pair, session, local_day, decision_time, mode, break_side, trade_side,
        rail_low, rail_high, rail_range, displacement, geometry,
        path_efficiency, settle, component, volume_count,
        usd_component=usd_component,
    )


def build_observations(
    bars_by_pair: dict[str, list[Bar]], start: int, end: int
) -> tuple[list[Observation], dict]:
    maps = {pair: {bar.time: bar for bar in bars} for pair, bars in bars_by_pair.items()}
    start_day = dt.datetime.fromtimestamp(start, tz=UTC).date() - dt.timedelta(days=1)
    end_day = dt.datetime.fromtimestamp(end, tz=UTC).date() + dt.timedelta(days=1)
    observations: list[Observation] = []
    counters = Counter()
    day = start_day
    by_session_day: dict[tuple[str, str], list[Observation]] = defaultdict(list)
    while day <= end_day:
        for session in ("LONDON_MIDDAY", "LONDON_FIX"):
            reference_times, event_times = schedule(session, day)
            decision_time = event_times[-1] + BAR_SECONDS
            if not start <= decision_time < end:
                continue
            counters["scheduled_pair_sessions"] += len(bars_by_pair)
            for pair in bars_by_pair:
                lookup = maps[pair]
                if any(value not in lookup for value in reference_times + event_times):
                    counters["gap_pair_sessions"] += 1
                    continue
                observation = make_observation(
                    pair,
                    session,
                    day.isoformat(),
                    [lookup[value] for value in reference_times],
                    [lookup[value] for value in event_times],
                )
                observations.append(observation)
                by_session_day[(session, day.isoformat())].append(observation)
                counters["valid_pair_sessions"] += 1
                if observation.ambiguous:
                    counters["ambiguous_both_rail"] += 1
                elif observation.mode:
                    counters[f"structural_{observation.mode}"] += 1
        day += dt.timedelta(days=1)
    for rows in by_session_day.values():
        if len(rows) != 3 or {row.pair for row in rows} != set(bars_by_pair):
            counters["breadth_incomplete_session_days"] += 1
            continue
        denominator = sum(abs(row.usd_component) for row in rows)
        total = sum(row.usd_component for row in rows)
        breadth = abs(total) / (denominator + EPSILON)
        common_sign = 1 if total > 0 else -1 if total < 0 else 0
        for row in rows:
            row.breadth = breadth
            row.common_usd_sign = common_sign
        counters["breadth_complete_session_days"] += 1
    return observations, dict(sorted(counters.items()))


def derive_calibration(observations: list[Observation], prereg: dict) -> dict:
    thresholds: dict = {"structure": {}, "breadth": {}, "activity": {}}
    minimum_structure = prereg["calibration"][
        "minimum_structural_events_per_session_mode"
    ]
    for session in ("LONDON_MIDDAY", "LONDON_FIX"):
        thresholds["structure"][session] = {}
        for mode in ("ACCEPT_CONTINUATION", "REJECT_FADE"):
            rows = [row for row in observations if row.session == session and row.mode == mode]
            if len(rows) < minimum_structure:
                raise ValueError(f"CALIBRATION_STRUCTURE_MINIMUM_FAILED:{session}:{mode}:{len(rows)}")
            thresholds["structure"][session][mode] = {
                "rows": len(rows),
                "D_Q50": max(1.0, quantile([row.displacement for row in rows], 0.50)),
                "D_Q67": max(1.0, quantile([row.displacement for row in rows], 0.67)),
                "G_Q50": quantile([row.geometry for row in rows], 0.50),
                "G_Q67": quantile([row.geometry for row in rows], 0.67),
            }
        breadth_values: dict[tuple[str, str], float] = {}
        for row in observations:
            if row.session == session and row.breadth is not None:
                breadth_values[(row.session, row.local_day)] = row.breadth
        values = list(breadth_values.values())
        if len(values) < prereg["calibration"]["minimum_common_breadth_days_per_session"]:
            raise ValueError(f"CALIBRATION_BREADTH_MINIMUM_FAILED:{session}:{len(values)}")
        thresholds["breadth"][session] = {
            "days": len(values),
            "Q50": quantile(values, 0.50),
        }
    for pair in prereg["input"]["symbols"]:
        thresholds["activity"][pair] = {}
        for session in ("LONDON_MIDDAY", "LONDON_FIX"):
            rows = [
                row for row in observations if row.pair == pair and row.session == session
            ]
            minimum = prereg["calibration"][
                "minimum_valid_activity_days_per_pair_session"
            ]
            if len(rows) < minimum:
                raise ValueError(f"CALIBRATION_ACTIVITY_MINIMUM_FAILED:{pair}:{session}:{len(rows)}")
            median_volume = statistics.median(row.volume_count for row in rows)
            if median_volume <= 0:
                raise ValueError(f"CALIBRATION_ACTIVITY_MEDIAN_INVALID:{pair}:{session}")
            values = [row.volume_count / (median_volume + EPSILON) for row in rows]
            thresholds["activity"][pair][session] = {
                "days": len(rows),
                "median_event_price_count": median_volume,
                "A_Q25": quantile(values, 0.25),
                "A_Q50": quantile(values, 0.50),
            }
    attach_activity(observations, thresholds)
    return thresholds


def attach_activity(observations: list[Observation], calibration: dict) -> None:
    for row in observations:
        median = calibration["activity"][row.pair][row.session][
            "median_event_price_count"
        ]
        row.activity = row.volume_count / (median + EPSILON)


def qualifies(row: Observation, config: dict, calibration: dict) -> bool:
    if row.session != config["session"] or row.mode != config["mode"]:
        return False
    threshold = calibration["structure"][row.session][row.mode]
    if row.displacement < threshold["D_" + config["displacement_quantile"]]:
        return False
    if row.geometry < threshold["G_" + config["geometry_quantile"]]:
        return False
    if config["breadth"] == "MODE_MATCHED":
        if row.breadth is None:
            return False
        breadth_q50 = calibration["breadth"][row.session]["Q50"]
        if row.mode == "ACCEPT_CONTINUATION":
            pair_usd_break = (1 if row.pair == "USD_JPY" else -1) * row.break_side
            if row.breadth < breadth_q50 or pair_usd_break != row.common_usd_sign:
                return False
        elif row.breadth >= breadth_q50:
            return False
    if config["activity"] == "MODE_MATCHED":
        activity = calibration["activity"][row.pair][row.session]
        if row.activity is None:
            return False
        if row.mode == "ACCEPT_CONTINUATION":
            if row.activity < activity["A_Q50"]:
                return False
        elif not activity["A_Q25"] <= row.activity < activity["A_Q50"]:
            return False
    return True


def _price(
    pair: str, bar: Bar, side: str, field: str, entry: bool, slippage: float
) -> float:
    index = FIELDS[field]
    pip = pip_size(pair)
    if entry:
        observed = bar.ask[index] if side == "LONG" else bar.bid[index]
        return observed + slippage * pip if side == "LONG" else observed - slippage * pip
    observed = bar.bid[index] if side == "LONG" else bar.ask[index]
    return observed - slippage * pip if side == "LONG" else observed + slippage * pip


def _jpy_pnl(
    pair: str,
    side: str,
    entry: float,
    exit_: float,
    units: int,
    usd_jpy_rate: float | None,
) -> float | None:
    direction = 1.0 if side == "LONG" else -1.0
    quote_pnl = direction * (exit_ - entry) * units
    if pair == "USD_JPY":
        return quote_pnl
    if usd_jpy_rate is None:
        return None
    return quote_pnl * usd_jpy_rate


def _path_metrics(pair: str, side: str, entry_mid: float, path: list[Bar]) -> tuple[float, float]:
    if side == "LONG":
        favorable = max(bar.mid("h") for bar in path)
        adverse = min(bar.mid("l") for bar in path)
        mfe = (favorable - entry_mid) / pip_size(pair)
        mae = (adverse - entry_mid) / pip_size(pair)
    else:
        favorable = min(bar.mid("l") for bar in path)
        adverse = max(bar.mid("h") for bar in path)
        mfe = (entry_mid - favorable) / pip_size(pair)
        mae = (entry_mid - adverse) / pip_size(pair)
    return mfe, mae


def make_trade(
    row: Observation,
    config: dict,
    maps: dict[str, dict[int, Bar]],
    period_end: int,
    prereg: dict,
) -> tuple[dict | None, str | None]:
    prereg_sha256 = EXPECTED_PREREG_FILE_SHA256
    pair_map = maps[row.pair]
    # The open at decision time belongs to a bar whose outcome was not known at
    # decision.  The fixed contract therefore enters one exact M5 open later.
    entry_time = row.decision_time + BAR_SECONDS
    horizon = config["horizon_bars"]
    scheduled_exit_time = entry_time + horizon * BAR_SECONDS
    terminal = scheduled_exit_time >= period_end
    if entry_time not in pair_map:
        return None, "ENTRY_GAP"
    if terminal:
        candidates = [time for time in pair_map if entry_time <= time < period_end]
        if not candidates:
            return None, "TERMINAL_UNPRICEABLE"
        price_time = max(candidates)
        exit_field = "c"
        exit_time = price_time + BAR_SECONDS
    else:
        price_time = scheduled_exit_time
        exit_field = "o"
        exit_time = price_time
    if price_time not in pair_map:
        return None, "EXIT_GAP"
    expected_times = list(range(entry_time, price_time + 1, BAR_SECONDS))
    if any(time not in pair_map for time in expected_times):
        return None, "PATH_GAP"
    conversion = maps["USD_JPY"].get(price_time)
    if conversion is None:
        return None, "JPY_CONVERSION_GAP"
    conversion_rate = conversion.mid(exit_field)
    entry_bar = pair_map[entry_time]
    exit_bar = pair_map[price_time]
    entry_mid = entry_bar.mid("o")
    exit_mid = exit_bar.mid(exit_field)
    raw_pips = signed_pips(row.pair, row.trade_side, entry_mid, exit_mid)
    base_slippage = prereg["execution_arms"]["base_slippage_pips_per_side"]
    adverse_slippage = prereg["execution_arms"]["adverse_slippage_pips_per_side"]
    base_entry = _price(
        row.pair, entry_bar, row.trade_side, "o", True, base_slippage
    )
    base_exit = _price(
        row.pair, exit_bar, row.trade_side, exit_field, False, base_slippage
    )
    adverse_entry = _price(
        row.pair, entry_bar, row.trade_side, "o", True, adverse_slippage
    )
    adverse_exit = _price(
        row.pair, exit_bar, row.trade_side, exit_field, False, adverse_slippage
    )
    base_pips = signed_pips(row.pair, row.trade_side, base_entry, base_exit)
    adverse_pips = signed_pips(row.pair, row.trade_side, adverse_entry, adverse_exit)
    units = prereg["execution_arms"]["units"]
    raw_jpy = _jpy_pnl(
        row.pair, row.trade_side, entry_mid, exit_mid, units, conversion_rate
    )
    base_jpy = _jpy_pnl(
        row.pair, row.trade_side, base_entry, base_exit, units, conversion_rate
    )
    adverse_jpy = _jpy_pnl(
        row.pair, row.trade_side, adverse_entry, adverse_exit, units, conversion_rate
    )
    if any(value is None or not math.isfinite(value) for value in (raw_jpy, base_jpy, adverse_jpy)):
        return None, "JPY_CONVERSION_INVALID"
    path = [pair_map[time] for time in expected_times]
    mfe, mae = _path_metrics(row.pair, row.trade_side, entry_mid, path)
    latency_time = entry_time + (
        prereg["execution_arms"]["latency_sensitivity_bars"] * BAR_SECONDS
    )
    latency_exit_time = latency_time + horizon * BAR_SECONDS
    latency_raw_pips = None
    if latency_exit_time < period_end and latency_time in pair_map and latency_exit_time in pair_map:
        latency_times = list(range(latency_time, latency_exit_time + 1, BAR_SECONDS))
        if all(time in pair_map for time in latency_times):
            latency_raw_pips = signed_pips(
                row.pair,
                row.trade_side,
                pair_map[latency_time].mid("o"),
                pair_map[latency_exit_time].mid("o"),
            )
    feature = {
        "pair": row.pair,
        "session": row.session,
        "local_day": row.local_day,
        "decision_time": iso_utc(row.decision_time),
        "mode": row.mode,
        "break_side": row.break_side,
        "trade_side": row.trade_side,
        "rail_log_range": row.rail_log_range,
        "displacement": row.displacement,
        "geometry": row.geometry,
        "breadth": row.breadth,
        "activity": row.activity,
    }
    feature_sha256 = sha_bytes(canonical(feature))
    signal_id = sha_bytes(
        canonical(
            {
                "candidate_id": "FX_SESSION_BREAK_RESPONSE_SURFACE_V5",
                "preregistration_sha256": prereg_sha256,
                "feature_sha256": feature_sha256,
                "pair": row.pair,
                "decision_time": iso_utc(row.decision_time),
                "mode": row.mode,
                "side": row.trade_side,
            }
        )
    )
    lineage_id = sha_bytes(
        canonical(
            {
                "signal_id": signal_id,
                "config_id": config["config_id"],
                "entry_time": iso_utc(entry_time),
                "exit_time": iso_utc(exit_time),
            }
        )
    )
    return {
        "signal_id": signal_id,
        "lineage_id": lineage_id,
        "feature_sha256": feature_sha256,
        "config_id": config["config_id"],
        "pair": row.pair,
        "session": row.session,
        "mode": row.mode,
        "side": row.trade_side,
        "decision_time": row.decision_time,
        "entry_time": entry_time,
        "exit_time": exit_time,
        "price_time": price_time,
        "horizon_bars": horizon,
        "terminal_liquidation": terminal,
        "raw_pips": raw_pips,
        "base_pips": base_pips,
        "adverse_pips": adverse_pips,
        "raw_jpy": raw_jpy,
        "base_jpy": base_jpy,
        "adverse_jpy": adverse_jpy,
        "mfe_pips": mfe,
        "mae_pips": mae,
        "latency_plus_5m_raw_pips": latency_raw_pips,
        "notional_jpy": units * entry_mid * conversion_rate
        if row.pair != "USD_JPY" else units * entry_mid,
    }, None


def evaluate_config(
    config: dict,
    observations: list[Observation],
    bars_by_pair: dict[str, list[Bar]],
    calibration: dict,
    split_start: int,
    split_end: int,
    prereg: dict,
) -> tuple[list[dict], dict]:
    maps = {pair: {bar.time: bar for bar in bars} for pair, bars in bars_by_pair.items()}
    trades = []
    counters = Counter()
    for row in observations:
        if not qualifies(row, config, calibration):
            continue
        counters["raw_qualified_signals"] += 1
        trade, reason = make_trade(row, config, maps, split_end, prereg)
        if trade is None:
            counters[reason] += 1
            continue
        trades.append(trade)
    trades.sort(key=lambda row: (row["exit_time"], row["entry_time"], row["signal_id"]))
    return trades, dict(sorted(counters.items()))


def _max_realized_drawdown(trades: list[dict], field: str, initial: float) -> tuple[float, bool]:
    equity = initial
    peak = initial
    worst = 0.0
    ruin = False
    for trade in sorted(trades, key=lambda row: (row["exit_time"], row["lineage_id"])):
        equity += trade[field]
        peak = max(peak, equity)
        if peak > 0:
            worst = min(worst, (equity - peak) / peak)
        ruin = ruin or equity <= 0
    return worst, ruin


def _max_gross_leverage(trades: list[dict], initial_equity: float) -> float:
    events = []
    for trade in trades:
        events.append((trade["entry_time"], 1, trade["notional_jpy"]))
        events.append((trade["exit_time"], -1, trade["notional_jpy"]))
    gross = maximum = 0.0
    # Exits sort before entries at one timestamp, matching fixed-open turnover.
    for _, kind, value in sorted(events, key=lambda row: (row[0], row[1])):
        gross += kind * value
        maximum = max(maximum, gross / initial_equity)
    return maximum


def density_metrics(trades: list[dict]) -> dict:
    by_pair = Counter(trade["pair"] for trade in trades)
    days = {dt.datetime.fromtimestamp(trade["decision_time"], tz=UTC).date().isoformat()
            for trade in trades}
    daily_counts = Counter(
        dt.datetime.fromtimestamp(trade["decision_time"], tz=UTC).date().isoformat()
        for trade in trades
    )
    weights = list(daily_counts.values())
    n_eff = (sum(weights) ** 2 / sum(value * value for value in weights)) if weights else 0.0
    return {
        "trades": len(trades),
        "active_utc_days": len(days),
        "trades_by_pair": dict(sorted(by_pair.items())),
        "pairs_with_24_trades": sum(value >= 24 for value in by_pair.values()),
        "pairs_with_12_trades": sum(value >= 12 for value in by_pair.values()),
        "N_eff_utc_day_kish": n_eff,
    }


def summarize(
    trades: list[dict], split_start: int, split_end: int, initial_equity: float,
    gross_leverage_cap: float,
) -> dict:
    density = density_metrics(trades)
    pair_means = {}
    for pair in ("EUR_USD", "USD_JPY", "AUD_USD"):
        values = [trade["raw_pips"] for trade in trades if trade["pair"] == pair]
        pair_means[pair] = sum(values) / len(values) if values else None
    month_values: dict[str, list[float]] = defaultdict(list)
    for trade in trades:
        month_values[anchored_month(trade["decision_time"], split_start)].append(
            trade["raw_pips"]
        )
    anchored_count = 0
    cursor = dt.datetime.fromtimestamp(split_start, tz=UTC)
    while int(add_months(cursor, anchored_count).timestamp()) < split_end:
        anchored_count += 1
    month_means = {
        f"M{index:02d}": (
            sum(month_values.get(f"M{index:02d}", []))
            / len(month_values.get(f"M{index:02d}", []))
            if month_values.get(f"M{index:02d}") else None
        )
        for index in range(1, anchored_count + 1)
    }
    arm_metrics = {}
    for arm, pips_field, jpy_field in (
        ("RAW_SIGNAL", "raw_pips", "raw_jpy"),
        ("EXECUTABLE_BASE", "base_pips", "base_jpy"),
        ("ADVERSE_STRESS", "adverse_pips", "adverse_jpy"),
    ):
        pips = [trade[pips_field] for trade in trades]
        jpy = [trade[jpy_field] for trade in trades]
        drawdown, ruin = _max_realized_drawdown(trades, jpy_field, initial_equity)
        arm_metrics[arm] = {
            "mean_pips": sum(pips) / len(pips) if pips else None,
            "total_pips": sum(pips),
            "total_jpy": sum(jpy),
            "equity_multiple": (initial_equity + sum(jpy)) / initial_equity,
            "max_realized_drawdown_pct": drawdown * 100.0,
            "ruin": ruin,
        }
    raw = [trade["raw_pips"] for trade in trades]
    base = [trade["base_pips"] for trade in trades]
    latency = [trade["latency_plus_5m_raw_pips"] for trade in trades
               if trade["latency_plus_5m_raw_pips"] is not None]
    terminal = [trade for trade in trades if trade["terminal_liquidation"]]
    max_leverage = _max_gross_leverage(trades, initial_equity)
    return {
        "density": density,
        "arms": arm_metrics,
        "direction_accuracy": sum(value > 0 for value in raw) / len(raw) if raw else None,
        "raw_mfe_mean_pips": (
            sum(trade["mfe_pips"] for trade in trades) / len(trades) if trades else None
        ),
        "raw_mae_mean_pips": (
            sum(trade["mae_pips"] for trade in trades) / len(trades) if trades else None
        ),
        "pair_raw_mean_pips": pair_means,
        "positive_pairs": sum(value is not None and value > 0 for value in pair_means.values()),
        "anchored_month_raw_mean_pips": month_means,
        "positive_anchored_months": sum(
            value is not None and value > 0 for value in month_means.values()
        ),
        "positive_anchored_month_fraction": (
            sum(value is not None and value > 0 for value in month_means.values())
            / len(month_means) if month_means else 0.0
        ),
        "cost_drag_mean_pips": (
            sum(r - b for r, b in zip(raw, base)) / len(raw) if raw else None
        ),
        "break_even_round_trip_cost_c_star_pips": sum(raw) / len(raw) if raw else None,
        "latency_plus_5m": {
            "scorable_trades": len(latency),
            "raw_mean_pips": sum(latency) / len(latency) if latency else None,
        },
        "terminal_liquidations": len(terminal),
        "terminal_mtm_jpy": {
            "RAW_SIGNAL": sum(trade["raw_jpy"] for trade in terminal),
            "EXECUTABLE_BASE": sum(trade["base_jpy"] for trade in terminal),
            "ADVERSE_STRESS": sum(trade["adverse_jpy"] for trade in terminal),
        },
        "terminal_open_inventory_count_after_liquidation": 0,
        "max_observed_gross_leverage": max_leverage,
        "gross_leverage_cap": gross_leverage_cap,
        "gross_leverage_guard_breaches": int(max_leverage > gross_leverage_cap),
        "nonfinite_accounting": any(
            not math.isfinite(trade[field])
            for trade in trades
            for field in ("raw_pips", "base_pips", "adverse_pips", "raw_jpy", "base_jpy", "adverse_jpy")
        ),
    }


def calendar_days(start: int, end: int) -> list[str]:
    first = dt.datetime.fromtimestamp(start, tz=UTC).date()
    last = dt.datetime.fromtimestamp(end - 1, tz=UTC).date()
    values = []
    while first <= last:
        values.append(first.isoformat())
        first += dt.timedelta(days=1)
    return values


def daily_matrices(
    configs: list[dict], trades_by_config: dict[str, list[dict]], days: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    day_index = {day: index for index, day in enumerate(days)}
    sums = np.zeros((len(configs), len(days)), dtype=np.float64)
    counts = np.zeros((len(configs), len(days)), dtype=np.float64)
    for config_index, config in enumerate(configs):
        for trade in trades_by_config[config["config_id"]]:
            day = dt.datetime.fromtimestamp(trade["decision_time"], tz=UTC).date().isoformat()
            index = day_index[day]
            sums[config_index, index] += trade["raw_pips"]
            counts[config_index, index] += 1.0
    return sums, counts


def common_block_weights(
    day_count: int, resamples: int, seed: int, block: int
) -> np.ndarray:
    if day_count < block:
        raise ValueError("five-day block bootstrap requires at least five days")
    rng = np.random.default_rng(seed)
    blocks_per = int(math.ceil(day_count / block))
    starts = rng.integers(0, day_count - block + 1, size=(resamples, blocks_per))
    offsets = np.arange(block, dtype=np.int64)
    indices = (starts[:, :, None] + offsets).reshape(resamples, -1)[:, :day_count]
    weights = np.zeros((resamples, day_count), dtype=np.float64)
    row_ids = np.repeat(np.arange(resamples), day_count)
    np.add.at(weights, (row_ids, indices.reshape(-1)), 1.0)
    return weights


def max_t_lcbs(
    configs: list[dict], sums: np.ndarray, counts: np.ndarray, weights: np.ndarray
) -> tuple[dict[str, float], float, dict]:
    family_count = len(configs)
    if family_count != 128 or sums.shape[0] != 128 or counts.shape[0] != 128:
        raise ValueError("MAX_T_REQUIRES_EXACTLY_128_CONFIG_COLUMNS")
    total_sums = sums.sum(axis=1)
    total_counts = counts.sum(axis=1)
    if not np.all(np.isfinite(total_counts) & (total_counts > 0)):
        failed = [
            configs[index]["config_id"] for index in range(128)
            if not math.isfinite(float(total_counts[index])) or total_counts[index] <= 0
        ]
        raise ValueError(
            "MAX_T_ALL_128_STANDARDIZATION_FAILED:" + ",".join(failed)
        )
    observed = np.divide(
        total_sums, total_counts, out=np.full_like(total_sums, np.nan),
        where=total_counts > 0,
    )
    boot_sums = weights @ sums.T
    boot_counts = weights @ counts.T
    if not np.all(np.isfinite(boot_counts) & (boot_counts > 0)):
        failed_columns = np.where(
            ~np.all(np.isfinite(boot_counts) & (boot_counts > 0), axis=0)
        )[0]
        failed = [configs[int(index)]["config_id"] for index in failed_columns]
        raise ValueError(
            "MAX_T_ALL_128_STANDARDIZATION_FAILED:" + ",".join(failed)
        )
    boot_means = np.divide(
        boot_sums, boot_counts, out=np.full_like(boot_sums, np.nan),
        where=boot_counts > 0,
    )
    standard_errors = np.nanstd(boot_means, axis=0, ddof=1)
    standardized = (
        np.isfinite(observed)
        & np.isfinite(standard_errors)
        & (standard_errors > 0)
        & np.all(np.isfinite(boot_means), axis=0)
    )
    standardized_count = int(standardized.sum())
    if standardized_count != 128:
        failed = [
            configs[index]["config_id"] for index in range(128)
            if not standardized[index]
        ]
        raise ValueError(
            "MAX_T_ALL_128_STANDARDIZATION_FAILED:" + ",".join(failed)
        )
    t_values = (boot_means - observed) / standard_errors
    if t_values.shape != (weights.shape[0], 128) or not np.all(np.isfinite(t_values)):
        raise ValueError("MAX_T_ALL_128_FINITE_T_STATISTICS_REQUIRED")
    critical = float(np.quantile(np.nanmax(t_values, axis=1), 0.95, method="linear"))
    if not math.isfinite(critical):
        raise ValueError("MAX_T_CRITICAL_VALUE_NONFINITE")
    lcbs: dict[str, float] = {}
    for index, config in enumerate(configs):
        lcbs[config["config_id"]] = float(
            observed[index] - critical * standard_errors[index]
        )
    return lcbs, critical, {
        "family_count": family_count,
        "standardized_count": standardized_count,
        "common_resamples": int(weights.shape[0]),
        "finite_t_statistics": int(np.isfinite(t_values).sum()),
        "expected_finite_t_statistics": int(weights.shape[0] * 128),
    }


def percentile_lcb(values: np.ndarray, weights: np.ndarray) -> float | None:
    if values.ndim != 1 or values.shape[0] != weights.shape[1]:
        raise ValueError("bootstrap value/day dimension mismatch")
    means = (weights @ values) / weights.sum(axis=1)
    return float(np.quantile(means, 0.05, method="linear"))


def density_pass(density: dict, floor: dict, pair_key: str) -> bool:
    return (
        density["trades"] >= floor["trades"]
        and density["active_utc_days"] >= floor["active_utc_days"]
        and density[pair_key] >= floor["pairs_meeting_floor"]
    )


def select_winner(
    configs: list[dict], summaries: dict[str, dict], lcbs: dict[str, float | None],
    prereg: dict,
) -> tuple[dict, list[dict]]:
    floor = prereg["selection"]["density_floor"]
    candidates = []
    for config in configs:
        if not config["selection_eligible"]:
            continue
        summary = summaries[config["config_id"]]
        density = summary["density"]
        lcb = lcbs[config["config_id"]]
        admitted_density = density_pass(density, floor, "pairs_with_24_trades")
        pair_values = list(summary["pair_raw_mean_pips"].values())
        candidates.append(
            {
                "config_id": config["config_id"],
                "density_pass": admitted_density,
                "raw_corrected_lcb_pips": lcb,
                "raw_mean_pips": summary["arms"]["RAW_SIGNAL"]["mean_pips"],
                "worst_pair_raw_mean_pips": (
                    min(pair_values) if all(value is not None for value in pair_values)
                    else None
                ),
                "positive_anchored_month_fraction": summary[
                    "positive_anchored_month_fraction"
                ],
                "N_eff": density["N_eff_utc_day_kish"],
            }
        )
    eligible = [row for row in candidates if row["density_pass"] and row["raw_corrected_lcb_pips"] is not None]
    if not eligible:
        raise ValueError("DISCOVERY_SELECTION_NO_DENSITY_ELIGIBLE_CONFIG")
    eligible.sort(
        key=lambda row: (
            -row["raw_corrected_lcb_pips"],
            -row["raw_mean_pips"],
            -(row["worst_pair_raw_mean_pips"]
              if row["worst_pair_raw_mean_pips"] is not None else -math.inf),
            -row["positive_anchored_month_fraction"],
            -row["N_eff"],
            row["config_id"],
        )
    )
    winner_id = eligible[0]["config_id"]
    winner = next(config for config in configs if config["config_id"] == winner_id)
    return winner, candidates


def exact_ablation(config: dict, configs: list[dict]) -> dict:
    matches = [
        row for row in configs
        if row["session"] == config["session"]
        and row["mode"] == config["mode"]
        and row["displacement_quantile"] == config["displacement_quantile"]
        and row["geometry_quantile"] == config["geometry_quantile"]
        and row["horizon_bars"] == config["horizon_bars"]
        and row["breadth"] == "ANY"
        and row["activity"] == "ANY"
    ]
    if len(matches) != 1:
        raise ValueError("exact ANY/ANY ablation is not unique")
    return matches[0]


def interaction_daily_difference(
    winner_trades: list[dict], ablation_trades: list[dict], days: list[str],
    daily_capacity: int,
) -> np.ndarray:
    if daily_capacity <= 0:
        raise ValueError("interaction daily capacity must be positive")
    index = {day: position for position, day in enumerate(days)}
    values = np.zeros(len(days), dtype=np.float64)
    for sign, trades in ((1.0, winner_trades), (-1.0, ablation_trades)):
        for trade in trades:
            day = dt.datetime.fromtimestamp(trade["decision_time"], tz=UTC).date().isoformat()
            # Three preregistered pair slots are the fixed daily capacity for a
            # single session/config.  No-trade slots are therefore explicit 0.
            values[index[day]] += sign * trade["raw_pips"] / daily_capacity
    return values


def compact_config_result(summary: dict, counters: dict, lcb: float | None) -> dict:
    return {
        "raw_corrected_lcb_pips": lcb,
        "raw_mean_pips": summary["arms"]["RAW_SIGNAL"]["mean_pips"],
        "base_mean_pips": summary["arms"]["EXECUTABLE_BASE"]["mean_pips"],
        "adverse_mean_pips": summary["arms"]["ADVERSE_STRESS"]["mean_pips"],
        "cost_drag_mean_pips": summary["cost_drag_mean_pips"],
        "density": summary["density"],
        "positive_pairs": summary["positive_pairs"],
        "positive_anchored_month_fraction": summary[
            "positive_anchored_month_fraction"
        ],
        "execution_counters": counters,
    }


def run() -> dict:
    prereg = load_preregistration()
    prereg_checks = validate_preregistration(prereg)
    prereg_sha256 = sha_file(PREREG_PATH)
    bars_discovery, decode_audit = load_discovery_phase(prereg)
    calibration_start = parse_time(prereg["splits"]["calibration"]["from_utc"])
    calibration_end = parse_time(prereg["splits"]["calibration"]["to_utc"])
    discovery_start = parse_time(prereg["splits"]["discovery"]["from_utc"])
    discovery_end = parse_time(prereg["splits"]["discovery"]["to_utc"])
    calibration_observations, calibration_counters = build_observations(
        bars_discovery, calibration_start, calibration_end
    )
    calibration = derive_calibration(calibration_observations, prereg)
    discovery_observations, discovery_observation_counters = build_observations(
        bars_discovery, discovery_start, discovery_end
    )
    attach_activity(discovery_observations, calibration)
    configs = prereg["family"]["configs"]
    discovery_trades = {}
    discovery_summaries = {}
    discovery_execution_counters = {}
    initial_equity = prereg["portfolio_and_reporting"]["initial_equity_jpy"]
    gross_leverage_cap = prereg["portfolio_and_reporting"][
        "gross_leverage_observation_cap"
    ]
    for config in configs:
        trades, counters = evaluate_config(
            config, discovery_observations, bars_discovery, calibration,
            discovery_start, discovery_end, prereg,
        )
        discovery_trades[config["config_id"]] = trades
        discovery_summaries[config["config_id"]] = summarize(
            trades, discovery_start, discovery_end, initial_equity,
            gross_leverage_cap,
        )
        discovery_execution_counters[config["config_id"]] = counters
    discovery_days = calendar_days(discovery_start, discovery_end)
    daily_sums, daily_counts = daily_matrices(configs, discovery_trades, discovery_days)
    weights = common_block_weights(
        len(discovery_days),
        prereg["selection"]["bootstrap_resamples"],
        prereg["selection"]["bootstrap_seed"],
        prereg["selection"]["bootstrap_block_days"],
    )
    lcbs, max_t_critical, max_t_audit = max_t_lcbs(
        configs, daily_sums, daily_counts, weights
    )
    winner, selection_table = select_winner(configs, discovery_summaries, lcbs, prereg)
    ablation = exact_ablation(winner, configs)
    winner_lock = {
        "candidate_id": prereg["candidate_id"],
        "preregistration_sha256": prereg_sha256,
        "selected_config": winner,
        "exact_any_any_ablation_config_id": ablation["config_id"],
        "selection_source_exclusive_end_utc": prereg["splits"]["discovery"]["to_utc"],
        "validation_rows_decoded_at_lock": 0,
    }
    winner_lock_sha256 = sha_bytes(canonical(winner_lock))

    # Only now may the locked validation suffix be semantically decoded.
    validation_bars = load_validation_phase(prereg, decode_audit, winner_lock_sha256)
    validation_start = parse_time(
        prereg["splits"]["locked_internal_validation"]["from_utc"]
    )
    validation_end = parse_time(
        prereg["splits"]["locked_internal_validation"]["to_utc"]
    )
    validation_observations, validation_observation_counters = build_observations(
        validation_bars, validation_start, validation_end
    )
    attach_activity(validation_observations, calibration)
    winner_validation_trades, winner_validation_counters = evaluate_config(
        winner, validation_observations, validation_bars, calibration,
        validation_start, validation_end, prereg,
    )
    ablation_validation_trades, ablation_validation_counters = evaluate_config(
        ablation, validation_observations, validation_bars, calibration,
        validation_start, validation_end, prereg,
    )
    winner_validation_summary = summarize(
        winner_validation_trades, validation_start, validation_end,
        initial_equity, gross_leverage_cap,
    )
    ablation_validation_summary = summarize(
        ablation_validation_trades, validation_start, validation_end,
        initial_equity, gross_leverage_cap,
    )
    validation_days = calendar_days(validation_start, validation_end)
    validation_weights = common_block_weights(
        len(validation_days), prereg["selection"]["bootstrap_resamples"],
        prereg["selection"]["bootstrap_seed"] + 1,
        prereg["selection"]["bootstrap_block_days"],
    )
    winner_daily = np.zeros(len(validation_days), dtype=float)
    day_index = {day: index for index, day in enumerate(validation_days)}
    winner_daily_counts = np.zeros(len(validation_days), dtype=float)
    for trade in winner_validation_trades:
        day = dt.datetime.fromtimestamp(trade["decision_time"], tz=UTC).date().isoformat()
        winner_daily[day_index[day]] += trade["raw_pips"]
        winner_daily_counts[day_index[day]] += 1.0
    boot_sum = validation_weights @ winner_daily
    boot_count = validation_weights @ winner_daily_counts
    boot_mean = np.divide(
        boot_sum, boot_count, out=np.full_like(boot_sum, np.nan), where=boot_count > 0
    )
    validation_lcb = float(np.nanquantile(boot_mean, 0.05, method="linear"))
    interaction_values = interaction_daily_difference(
        winner_validation_trades, ablation_validation_trades, validation_days,
        len(prereg["input"]["symbols"]),
    )
    interaction_lcb = percentile_lcb(interaction_values, validation_weights)
    density = winner_validation_summary["density"]
    density_ok = density_pass(
        density, prereg["validation"]["density_floor"], "pairs_with_12_trades"
    )
    raw_mean = winner_validation_summary["arms"]["RAW_SIGNAL"]["mean_pips"]
    pass_checks = {
        "raw_mean_gt_zero": raw_mean is not None and raw_mean > 0,
        "raw_95pct_lcb_gt_zero": validation_lcb > 0,
        "interaction_lcb_gt_zero": interaction_lcb is not None and interaction_lcb > 0,
        "positive_pairs_gte_2": winner_validation_summary["positive_pairs"] >= 2,
        "positive_anchored_months_gte_2_of_3": (
            winner_validation_summary["positive_anchored_months"] >= 2
            and len(winner_validation_summary["anchored_month_raw_mean_pips"]) == 3
        ),
        "density": density_ok,
        "terminal_inventory_zero": winner_validation_summary[
            "terminal_open_inventory_count_after_liquidation"
        ] == 0,
        "no_ruin": not any(
            winner_validation_summary["arms"][arm]["ruin"] for arm in ARMS
        ),
        "no_nonfinite": not winner_validation_summary["nonfinite_accounting"],
        "no_gross_leverage_guard_breach": winner_validation_summary[
            "gross_leverage_guard_breaches"
        ] == 0,
        "leverage_fitting_false": prereg["portfolio_and_reporting"][
            "leverage_fitting"
        ] is False,
    }
    validation_passed = all(pass_checks.values())
    if not pass_checks["raw_mean_gt_zero"]:
        reason = "VALIDATION_RAW_EDGE_ABSENT"
    elif not pass_checks["raw_95pct_lcb_gt_zero"]:
        reason = "VALIDATION_RAW_EDGE_NOT_STATISTICALLY_POSITIVE"
    elif not pass_checks["interaction_lcb_gt_zero"]:
        reason = "MODE_MATCHED_INTERACTION_NOT_INCREMENTALLY_POSITIVE"
    elif not pass_checks["density"]:
        reason = "VALIDATION_DENSITY_FLOOR_FAILED"
    elif not validation_passed:
        reason = "VALIDATION_STABILITY_OR_HARD_GUARD_FAILED"
    else:
        reason = "LOCKED_INTERNAL_VALIDATION_PASSED_REQUIRES_FUTURE_HOLDOUT"
    result = {
        "schema": "QR_FX_SESSION_BREAK_RESPONSE_RESULT_V1",
        "candidate_id": prereg["candidate_id"],
        "status": "INTERNAL_VALIDATION_PASS_NOT_ADMITTED" if validation_passed
        else "REJECTED_INTERNAL_VALIDATION",
        "reason_code": reason,
        "profit_proven": False,
        "strategy_admitted": False,
        "future_holdout_required": True,
        "authority": {
            "network_attempts": 0,
            "credential_reads": 0,
            "external_order_attempts": 0,
            "external_orders": 0,
            "broker_mutations": 0,
            "launchd_actions": 0,
            "git_actions": 0,
        },
        "hashes": {
            "preregistration_sha256": prereg_sha256,
            "runner_sha256": sha_file(Path(__file__)),
            "dataset_sha256": prereg["input"]["canonical_dataset_sha256"],
            "winner_lock_sha256": winner_lock_sha256,
        },
        "decode_audit": decode_audit,
        "calibration": calibration,
        "calibration_observation_counters": calibration_counters,
        "discovery_observation_counters": discovery_observation_counters,
        "discovery": {
            "max_t_fwer_critical_value": max_t_critical,
            "max_t_audit": max_t_audit,
            "bootstrap_resamples": prereg["selection"]["bootstrap_resamples"],
            "selected_config": winner,
            "selected_summary": discovery_summaries[winner["config_id"]],
            "selected_execution_counters": discovery_execution_counters[
                winner["config_id"]
            ],
            "selected_corrected_lcb_pips": lcbs[winner["config_id"]],
            "exact_any_any_ablation_config": ablation,
            "config_results": {
                config["config_id"]: compact_config_result(
                    discovery_summaries[config["config_id"]],
                    discovery_execution_counters[config["config_id"]],
                    lcbs[config["config_id"]],
                ) for config in configs
            },
            "selection_table_32": selection_table,
        },
        "winner_lock": winner_lock,
        "locked_internal_validation": {
            "observation_counters": validation_observation_counters,
            "winner": winner_validation_summary,
            "winner_execution_counters": winner_validation_counters,
            "raw_95pct_lcb_pips": validation_lcb,
            "exact_any_any_ablation": ablation_validation_summary,
            "exact_any_any_ablation_execution_counters": ablation_validation_counters,
            "daily_capacity_normalized_interaction_mean_pips": float(
                np.mean(interaction_values)
            ),
            "daily_capacity_normalized_interaction_95pct_lcb_pips": interaction_lcb,
            "pass_checks": pass_checks,
            "passed": validation_passed,
        },
        "limitations": [
            "OANDA volume is price-count activity, not traded volume or true order flow.",
            "Financing is modeled as zero over the maximum four-hour vehicle.",
            "Drawdown is based on chronological realized/terminal exits, not intrabar portfolio mark-to-market.",
            "Internal validation is not untouched future evidence and cannot prove profit.",
        ],
    }
    result["result_sha256"] = sha_bytes(canonical(result))
    packet = {
        "schema": "QR_FX_SESSION_BREAK_RESPONSE_EVIDENCE_PACKET_V1",
        "candidate_id": prereg["candidate_id"],
        "status": result["status"],
        "reason_code": reason,
        "result_sha256": result["result_sha256"],
        "preregistration_sha256": prereg_sha256,
        "runner_sha256": result["hashes"]["runner_sha256"],
        "dataset_sha256": prereg["input"]["canonical_dataset_sha256"],
        "winner_lock_sha256": winner_lock_sha256,
        "selected_config_id": winner["config_id"],
        "validation_passed": validation_passed,
        "profit_proven": False,
        "strategy_admitted": False,
        "holdout_decoded_rows": 0,
        "external_orders": 0,
    }
    packet["packet_sha256"] = sha_bytes(canonical(packet))
    atomic_json(RESULT_PATH, result)
    atomic_json(PACKET_PATH, packet)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-once", action="store_true")
    args = parser.parse_args()
    if not args.run_once:
        parser.error("the sealed replay requires explicit --run-once")
    if RESULT_PATH.exists() or PACKET_PATH.exists():
        raise RuntimeError("ONE_SHOT_RESULT_ALREADY_EXISTS")
    result = run()
    validation = result["locked_internal_validation"]
    print(
        json.dumps(
            {
                "status": result["status"],
                "reason_code": result["reason_code"],
                "selected_config_id": result["winner_lock"]["selected_config"]["config_id"],
                "validation_raw_mean_pips": validation["winner"]["arms"]["RAW_SIGNAL"]["mean_pips"],
                "validation_raw_lcb_pips": validation["raw_95pct_lcb_pips"],
                "validation_interaction_lcb_pips": validation[
                    "daily_capacity_normalized_interaction_95pct_lcb_pips"
                ],
                "validation_trades": validation["winner"]["density"]["trades"],
                "result_sha256": result["result_sha256"],
                "external_orders": result["authority"]["external_orders"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
