#!/usr/bin/env python3
"""Deterministic, offline replay for M5_GENERIC_EMA_POST_ENTRY_LGAR_V4.

The module has no network, credential, broker, order, or service-management
surface.  It reads only byte-bounded pre-holdout prefixes of an immutable local
OANDA BID/ASK capture.  Entry proposals are cost-blind.  One RAW-driven policy
path fixes action/exit timestamps, and BASE/ADVERSE economics are applied only
afterward to that same lineage.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[2]
PREREG_PATH = ROOT / "PREREGISTRATION.json"
RESULT_PATH = ROOT / "result.json"
PACKET_PATH = ROOT / "evidence_packet.json"

# Five minutes is the immutable capture cadence.  Changing it changes the
# decision/fill chronology and requires a new preregistered candidate.
BAR_SECONDS = 5 * 60
# The tiny denominator prevents division by zero only on an exactly flat path;
# it is numeric protection, not a market threshold.
EPSILON = 1e-15
SCENARIOS = ("raw", "base", "adverse")
FIELD_INDEX = {"o": 0, "h": 1, "l": 2, "c": 3}


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def parse_time(value):
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


def iso_utc(seconds):
    value = dt.datetime.fromtimestamp(seconds, tz=dt.timezone.utc)
    return value.strftime("%Y-%m-%dT%H:%M:%S.000000Z")


def utc_day(seconds):
    return dt.datetime.fromtimestamp(seconds, tz=dt.timezone.utc).date().isoformat()


def utc_month(seconds):
    return dt.datetime.fromtimestamp(seconds, tz=dt.timezone.utc).strftime("%Y-%m")


def quantile(values, probability):
    if not values:
        raise ValueError("quantile requires values")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * probability
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def pip_size(pair):
    return 0.01 if pair.endswith("_JPY") else 0.0001


def signed_pips(pair, side, entry, exit_):
    sign = 1.0 if side == "LONG" else -1.0
    return sign * (exit_ - entry) / pip_size(pair)


def side_from_sign(value):
    if value > 0:
        return "LONG"
    if value < 0:
        return "SHORT"
    return None


def opposite(side):
    return "SHORT" if side == "LONG" else "LONG"


def usd_node_sign(pair, side):
    # USD is the base of USD_JPY and the quote of the two USD crosses.  This is
    # currency graph orientation, not a fitted parameter.
    base_usd = pair.startswith("USD_")
    return 1 if (side == "LONG") == base_usd else -1


@dataclass(slots=True)
class Bar:
    time: int
    bid: tuple
    ask: tuple

    def mid(self, field):
        index = FIELD_INDEX[field]
        return (self.bid[index] + self.ask[index]) / 2.0

    def executable(self, side, field):
        index = FIELD_INDEX[field]
        return self.bid[index] if side == "LONG" else self.ask[index]

    def entry_executable(self, side):
        return self.ask[0] if side == "LONG" else self.bid[0]


@dataclass(slots=True)
class Feature:
    index: int
    decision_time: int
    trend_side: str
    path_efficiency: float
    impulse_side: str | None
    rail_side: str | None
    rail_kind: str
    spread_pips: float
    slot: str
    usd_one_bar_sign: int
    usd_breadth: float = 0.0
    usd_breadth_count: int = 0


@dataclass(slots=True)
class Signal:
    signal_id: str
    feature_hash: str
    pair: str
    side: str
    decision_time: int
    feature_index: int


@dataclass(slots=True)
class PendingEntry:
    signal: Signal
    fill_time: int


@dataclass(slots=True)
class Position:
    trade_id: str
    policy_id: str
    signal_id: str
    pair: str
    side: str
    decision_time: int
    entry_time: int
    entry_mid: float
    entry_observed: float
    fixed_exit_time: int
    scheduled_exit_time: int
    exit_action_time: int
    exit_reason: str
    tp_pips: float
    peak_close_mfe_pips: float = 0.0
    raw_mfe_pips: float = 0.0
    raw_mae_pips: float = 0.0
    tp_reached: bool = False
    last_state: str = "NEUTRAL"
    last_state_time: int = -1


@dataclass(slots=True)
class Trade:
    trade_id: str
    signal_id: str
    policy_id: str
    pair: str
    side: str
    decision_time: int
    entry_time: int
    exit_action_time: int
    exit_time: int
    price_time: int
    gap_detection_time: int | None
    exit_reason: str
    holding_bars: int
    raw_pips: float
    base_pips: float
    adverse_pips: float
    raw_jpy: float | None
    base_jpy: float | None
    adverse_jpy: float | None
    raw_mfe_pips: float
    raw_mae_pips: float
    terminal_liquidation: bool


def load_preregistration(path=PREREG_PATH):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_preregistration(prereg):
    expected = {
        "P0": "TIME_EXIT_12",
        "P1": "TIME_EXIT_24",
        "P2": "TIME_EXIT_48",
        "P3": "CALIBRATION_RAW_CLOSE_MFE_Q40_TP_MAX24",
        "P4": "CALIBRATION_RAW_CLOSE_MFE_Q40_TP_MAX48",
        "P5": "FX_LGAR",
        "P6": "FX_LGAR_PLUS_50PCT_CLOSE_MFE_GIVEBACK",
        "P7": "FX_LGAR_PROFIT_LOCK_PLUS_USD_NODE_BASKET",
    }
    checks = {
        "candidate": prereg["candidate_id"]
        == "M5_GENERIC_EMA_POST_ENTRY_LGAR_V4",
        "policies": {
            key: value["name"] for key, value in prereg["policies"].items()
        }
        == expected,
        "family_size": prereg["selection"]["family_size"] == 8,
        "ema": prereg["source_signal"]["ema_fast_bars"] == 3
        and prereg["source_signal"]["ema_slow_bars"] == 12,
        "cost_gate": prereg["execution_arms"]["cost_gate"] is False,
        "slippage": prereg["execution_arms"]["EXECUTABLE_BASE"].endswith(
            "0.3 pip slippage per side"
        )
        and prereg["execution_arms"]["ADVERSE_STRESS"].endswith(
            "0.9 pip slippage per side"
        ),
        "units": prereg["portfolio"]["units_per_lot"] == 1000,
        "caps": prereg["portfolio"]["pair_lot_cap"] == 4
        and prereg["portfolio"]["absolute_net_usd_node_lot_cap"] == 4
        and prereg["portfolio"]["same_sign_usd_node_lot_cap"] == 4
        and prereg["portfolio"]["gross_lot_cap"] == 8
        and prereg["portfolio"]["gross_leverage_cap"] == 20,
        "margin_ledger": prereg["portfolio"][
            "shared_hard_execution_equity_ledger"
        ].startswith("EXECUTABLE_BASE")
        and prereg["portfolio"]["adverse_stress_margin_rule"].startswith(
            "record first"
        ),
        "holdout_locked": prereg["splits"]["untouched_holdout"]["use"]
        == "LOCKED_NOT_READ"
        and prereg["splits"]["untouched_holdout"][
            "price_or_volume_decode_allowed"
        ]
        is False,
        "final_run_tuning_only": prereg["splits"]["opened_development"][
            "use"
        ].startswith("INVALIDATED_DIAGNOSTIC_NOT_EVIDENCE")
        and set(prereg["inputs"]["final_tuning_only_prefix_contract"])
        == set(prereg["inputs"]["symbols"]),
        "authority": prereg["network_allowed"] is False
        and prereg["credential_access_allowed"] is False
        and prereg["broker_mutation_allowed"] is False
        and prereg["live_order_authority"] is False,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError("preregistration mismatch: " + ",".join(failed))
    return checks


def _read_exact_prefix(path, exclusive_offset, expected_sha):
    """Read exactly the preregistered non-holdout prefix and no later byte."""
    digest = hashlib.sha256()
    remaining = exclusive_offset
    chunks = []
    with Path(path).open("rb", buffering=0) as handle:
        while remaining:
            chunk = handle.read(min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError("source ended before preregistered boundary")
            chunks.append(chunk)
            digest.update(chunk)
            remaining -= len(chunk)
    if digest.hexdigest() != expected_sha:
        raise ValueError(f"pre-holdout prefix hash mismatch: {path}")
    return b"".join(chunks)


def _parse_bar(raw, pair, semantic_end):
    row = json.loads(raw)
    if row.get("schema") != "QR_OANDA_HISTORICAL_M5_BA_ROW_V1":
        raise ValueError("unexpected candle schema")
    if row.get("instrument") != pair or row.get("granularity") != "M5":
        raise ValueError("candle identity mismatch")
    if row.get("price_component") != "BA" or row.get("complete") is not True:
        raise ValueError("candle is not a complete BID/ASK bar")
    timestamp = parse_time(row.get("time_utc"))
    if timestamp >= semantic_end:
        raise AssertionError("locked post-tuning row reached semantic decoder")
    volume = row.get("volume")
    if isinstance(volume, bool) or not isinstance(volume, int) or volume < 0:
        raise ValueError("invalid pre-holdout OANDA price-count volume")
    sides = []
    for side_name in ("bid", "ask"):
        side = row.get(side_name)
        if not isinstance(side, dict) or set(side) != {"o", "h", "l", "c"}:
            raise ValueError("invalid BID/ASK OHLC shape")
        values = tuple(float(side[name]) for name in ("o", "h", "l", "c"))
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError("non-finite or non-positive price")
        if not values[2] <= min(values[0], values[3]) <= max(
            values[0], values[3]
        ) <= values[1]:
            raise ValueError("invalid OHLC geometry")
        sides.append(values)
    bid, ask = sides
    if any(bid[index] > ask[index] for index in range(4)):
        raise ValueError("crossed BID/ASK")
    return Bar(timestamp, bid, ask)


def load_inputs(prereg):
    dataset_root = REPO_ROOT / prereg["inputs"]["dataset_root"]
    manifest_path = dataset_root / "manifest.json"
    gap_path = dataset_root / "gap_report.json"
    if sha_file(manifest_path) != prereg["inputs"]["manifest_sha256"]:
        raise ValueError("manifest bytes changed")
    if sha_file(gap_path) != prereg["inputs"]["gap_report_sha256"]:
        raise ValueError("gap report bytes changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("canonical_dataset_sha256")
        != prereg["inputs"]["canonical_dataset_sha256"]
    ):
        raise ValueError("dataset seal mismatch")
    if manifest.get("external_orders") != 0 or manifest.get(
        "external_order_attempts"
    ) != 0:
        raise ValueError("capture authority boundary mismatch")
    semantic_end = parse_time(prereg["splits"]["tuning"]["to_utc"])
    bars_by_pair = {}
    audit = {
        "dataset_manifest_sha256": prereg["inputs"]["manifest_sha256"],
        "canonical_dataset_sha256": prereg["inputs"][
            "canonical_dataset_sha256"
        ],
        "holdout_price_or_volume_rows_decoded": 0,
        "holdout_label_rows_computed": 0,
        "opened_development_price_or_volume_rows_decoded": 0,
        "opened_development_label_rows_computed": 0,
        "pairs": {},
    }
    for pair in prereg["inputs"]["symbols"]:
        contract = prereg["inputs"]["final_tuning_only_prefix_contract"][pair]
        payload = _read_exact_prefix(
            dataset_root / contract["path"],
            contract["exclusive_byte_offset"],
            contract["prefix_sha256"],
        )
        lines = payload.splitlines()
        if len(lines) != contract["prefix_rows"]:
            raise ValueError(f"prefix row count mismatch: {pair}")
        bars = [_parse_bar(line, pair, semantic_end) for line in lines]
        if any(bars[index].time >= bars[index + 1].time for index in range(len(bars) - 1)):
            raise ValueError(f"non-increasing candle timestamps: {pair}")
        bars_by_pair[pair] = bars
        audit["pairs"][pair] = {
            "bytes_read": len(payload),
            "rows_decoded": len(bars),
            "prefix_sha256": sha_bytes(payload),
            "first_time_utc": iso_utc(bars[0].time),
            "last_time_utc": iso_utc(bars[-1].time),
            "max_decoded_time_before_tuning_end": bars[-1].time < semantic_end,
        }
    return bars_by_pair, audit


def compute_features(pair, bars, prereg):
    fast_window = prereg["source_signal"]["ema_fast_bars"]
    slow_window = prereg["source_signal"]["ema_slow_bars"]
    return_window = prereg["calibration"]["path_efficiency_window_returns"]
    rail_bars = prereg["calibration"]["prior_rail_bars"]
    alpha_fast = 2.0 / (fast_window + 1.0)
    alpha_slow = 2.0 / (slow_window + 1.0)
    closes = [bar.mid("c") for bar in bars]
    highs = [bar.mid("h") for bar in bars]
    lows = [bar.mid("l") for bar in bars]
    features = [None] * len(bars)
    ema_fast = None
    ema_slow = None
    segment_start = 0
    for index, bar in enumerate(bars):
        if index == 0 or bar.time - bars[index - 1].time != BAR_SECONDS:
            segment_start = index
            ema_fast = closes[index]
            ema_slow = closes[index]
        else:
            ema_fast = alpha_fast * closes[index] + (1.0 - alpha_fast) * ema_fast
            ema_slow = alpha_slow * closes[index] + (1.0 - alpha_slow) * ema_slow
        required = max(slow_window, return_window + 1, rail_bars + 2)
        if index - segment_start + 1 < required or ema_fast == ema_slow:
            continue
        returns = [
            math.log(closes[position] / closes[position - 1])
            for position in range(index - return_window + 1, index + 1)
        ]
        path_efficiency = abs(sum(returns)) / (
            sum(abs(value) for value in returns) + EPSILON
        )
        impulse_side = side_from_sign(sum(returns))
        trend_side = "LONG" if ema_fast > ema_slow else "SHORT"
        rail_slice_start = index - rail_bars - 1
        rail_slice_end = index - 1
        upper = max(highs[rail_slice_start:rail_slice_end])
        lower = min(lows[rail_slice_start:rail_slice_end])
        previous = index - 1
        rail_candidates = []
        if closes[previous] > upper and closes[index] > upper:
            rail_candidates.append(("ACCEPTANCE", "LONG"))
        if closes[previous] < lower and closes[index] < lower:
            rail_candidates.append(("ACCEPTANCE", "SHORT"))
        if highs[index] > upper and closes[index] <= upper:
            rail_candidates.append(("REJECTION", "SHORT"))
        if lows[index] < lower and closes[index] >= lower:
            rail_candidates.append(("REJECTION", "LONG"))
        if len(rail_candidates) == 1:
            rail_kind, rail_side = rail_candidates[0]
        elif not rail_candidates:
            rail_kind, rail_side = "NONE", None
        else:
            rail_kind, rail_side = "AMBIGUOUS", None
        one_bar_sign = 1 if closes[index] > closes[previous] else -1 if closes[index] < closes[previous] else 0
        usd_sign = one_bar_sign if pair.startswith("USD_") else -one_bar_sign
        decision_time = bar.time + BAR_SECONDS
        when = dt.datetime.fromtimestamp(decision_time, tz=dt.timezone.utc)
        slot = f"{when.hour:02d}:{when.minute:02d}"
        features[index] = Feature(
            index=index,
            decision_time=decision_time,
            trend_side=trend_side,
            path_efficiency=path_efficiency,
            impulse_side=impulse_side,
            rail_side=rail_side,
            rail_kind=rail_kind,
            spread_pips=(bar.ask[3] - bar.bid[3]) / pip_size(pair),
            slot=slot,
            usd_one_bar_sign=usd_sign,
        )
    return features


def attach_usd_breadth(features_by_pair):
    by_time = defaultdict(list)
    for features in features_by_pair.values():
        for feature in features:
            if feature is not None:
                by_time[feature.decision_time].append(feature.usd_one_bar_sign)
    for features in features_by_pair.values():
        for feature in features:
            if feature is None:
                continue
            values = by_time[feature.decision_time]
            nonzero = [value for value in values if value]
            feature.usd_breadth_count = len(nonzero)
            feature.usd_breadth = (
                sum(nonzero) / len(nonzero) if len(nonzero) >= 2 else 0.0
            )


def make_signals(bars_by_pair, features_by_pair, prereg_sha):
    signals_by_time = defaultdict(list)
    pair_order = {pair: index for index, pair in enumerate(bars_by_pair)}
    for pair, bars in bars_by_pair.items():
        for feature in features_by_pair[pair]:
            if feature is None:
                continue
            bar = bars[feature.index]
            source_material = {
                "pair": pair,
                "decision_time": iso_utc(feature.decision_time),
                "mid_close": bar.mid("c"),
                "ema_side": feature.trend_side,
            }
            feature_hash = sha_bytes(canonical(source_material))
            signal_id = sha_bytes(
                canonical(
                    {
                        "candidate_id": "M5_GENERIC_EMA_POST_ENTRY_LGAR_V4",
                        "preregistration_sha256": prereg_sha,
                        "feature_sha256": feature_hash,
                        "pair": pair,
                        "decision_time": iso_utc(feature.decision_time),
                        "side": feature.trend_side,
                    }
                )
            )
            signals_by_time[feature.decision_time].append(
                Signal(
                    signal_id=signal_id,
                    feature_hash=feature_hash,
                    pair=pair,
                    side=feature.trend_side,
                    decision_time=feature.decision_time,
                    feature_index=feature.index,
                )
            )
    for signals in signals_by_time.values():
        signals.sort(key=lambda signal: (pair_order[signal.pair], signal.signal_id))
    return signals_by_time


def _exact_contiguous(bars, start_index, count):
    if start_index < 0 or start_index + count > len(bars):
        return False
    return all(
        bars[index + 1].time - bars[index].time == BAR_SECONDS
        for index in range(start_index, start_index + count - 1)
    )


def derive_calibration(
    bars_by_pair, features_by_pair, signals_by_time, prereg
):
    start = parse_time(prereg["splits"]["calibration"]["from_utc"])
    end = parse_time(prereg["splits"]["calibration"]["to_utc"])
    pe_q = prereg["calibration"]["path_efficiency_q"]
    spread_q = prereg["calibration"]["spread_quality_q"]
    mfe_q = prereg["calibration"]["closing_mfe_q"]
    horizon = prereg["calibration"]["closing_mfe_horizon_bars"]
    result = {"path_efficiency_q67": {}, "spread_slot_q75": {}, "mfe_q40_pips": {}}
    time_maps = {
        pair: {bar.time: index for index, bar in enumerate(bars)}
        for pair, bars in bars_by_pair.items()
    }
    for pair, features in features_by_pair.items():
        calibration_features = [
            feature
            for feature in features
            if feature is not None and start <= feature.decision_time < end
        ]
        pe_values = [feature.path_efficiency for feature in calibration_features]
        if len(pe_values) < 1000:
            raise ValueError(f"insufficient calibration path states: {pair}")
        result["path_efficiency_q67"][pair] = {
            "rows": len(pe_values),
            "value": quantile(pe_values, pe_q),
        }
        slots = defaultdict(list)
        for feature in calibration_features:
            slots[feature.slot].append(feature.spread_pips)
        if len(slots) != 288:
            raise ValueError(f"incomplete calibration UTC slots: {pair}")
        result["spread_slot_q75"][pair] = {}
        for slot, values in sorted(slots.items()):
            if len(values) < 40:
                raise ValueError(f"thin spread calibration slot: {pair}/{slot}")
            result["spread_slot_q75"][pair][slot] = {
                "rows": len(values),
                "value": quantile(values, spread_q),
            }
        by_side = {"LONG": [], "SHORT": []}
        for decision_time in sorted(time for time in signals_by_time if start <= time < end):
            for signal in signals_by_time[decision_time]:
                if signal.pair != pair:
                    continue
                # A completed-bar decision at T deliberately skips the open at
                # T; the first allowed open is the exact T+5m bar.
                entry_time = decision_time + BAR_SECONDS
                entry_index = time_maps[pair].get(entry_time)
                # The strict-later fill is two source bars after the signal
                # bar: the open at the decision boundary itself is forbidden.
                # Requiring an exact contiguous path from the source bar also
                # prevents a calibration label from jumping across a missing
                # M5 interval merely because T+5 happens to exist.
                if (
                    entry_index != signal.feature_index + 2
                    or not _exact_contiguous(
                        bars_by_pair[pair], signal.feature_index, horizon + 2
                    )
                ):
                    continue
                last_close_time = bars_by_pair[pair][entry_index + horizon - 1].time + BAR_SECONDS
                if last_close_time > end:
                    continue
                entry_mid = bars_by_pair[pair][entry_index].mid("o")
                values = [
                    signed_pips(
                        pair,
                        signal.side,
                        entry_mid,
                        bars_by_pair[pair][index].mid("c"),
                    )
                    for index in range(entry_index, entry_index + horizon)
                ]
                by_side[signal.side].append(max(0.0, max(values)))
        result["mfe_q40_pips"][pair] = {}
        for side, values in by_side.items():
            if len(values) < 500:
                raise ValueError(f"insufficient calibration MFE paths: {pair}/{side}")
            result["mfe_q40_pips"][pair][side] = {
                "rows": len(values),
                "value": quantile(values, mfe_q),
            }
    return result


class Market:
    def __init__(self, bars_by_pair, features_by_pair):
        self.bars_by_pair = bars_by_pair
        self.features_by_pair = features_by_pair
        self.open_index = {
            pair: {bar.time: index for index, bar in enumerate(bars)}
            for pair, bars in bars_by_pair.items()
        }
        self.close_index = {
            pair: {bar.time + BAR_SECONDS: index for index, bar in enumerate(bars)}
            for pair, bars in bars_by_pair.items()
        }

    def bar_open(self, pair, when):
        index = self.open_index[pair].get(when)
        return None if index is None else self.bars_by_pair[pair][index]

    def completed(self, pair, when):
        index = self.close_index[pair].get(when)
        if index is None:
            return None, None
        return self.bars_by_pair[pair][index], self.features_by_pair[pair][index]

    def usd_jpy_rate(self, when, field):
        if field == "o":
            bar = self.bar_open("USD_JPY", when)
            return None if bar is None else bar.mid("o")
        bar, _ = self.completed("USD_JPY", when)
        # Cross-pair PnL, equity, and leverage are unpriceable without an
        # exact-time USDJPY observation.  Never let a stale prior close alter a
        # hard guard or silently pass evidence completeness.
        return None if bar is None else bar.mid("c")

    def pips_to_jpy(self, pair, pips, when, field):
        units = 1000
        if pair.endswith("_JPY"):
            return pips * units * pip_size(pair)
        rate = self.usd_jpy_rate(when, field)
        if rate is None:
            return None
        return pips * units * pip_size(pair) * rate

    def notional_jpy(self, pair, when, field="c"):
        if field == "o":
            bar = self.bar_open(pair, when)
        else:
            bar, _ = self.completed(pair, when)
        if bar is None:
            return None
        if pair.startswith("USD_"):
            rate = self.usd_jpy_rate(when, field)
            return None if rate is None else 1000.0 * rate
        rate = self.usd_jpy_rate(when, field)
        return None if rate is None else 1000.0 * bar.mid(field) * rate


def policy_horizon(policy_id):
    return {
        "P0": 12,
        "P1": 24,
        "P2": 48,
        "P3": 24,
        "P4": 48,
        "P5": 48,
        "P6": 48,
        "P7": 48,
    }[policy_id]


class PolicySimulator:
    def __init__(
        self,
        policy_id,
        market,
        signals_by_time,
        calibration,
        prereg,
        split_name,
    ):
        self.policy_id = policy_id
        self.market = market
        self.signals_by_time = signals_by_time
        self.calibration = calibration
        self.prereg = prereg
        self.split_name = split_name
        split = prereg["splits"][split_name]
        self.start = parse_time(split["from_utc"])
        self.end = parse_time(split["to_utc"])
        self.open_positions = {}
        self.pending_entries = defaultdict(list)
        self.trades = []
        self.realized_jpy = {scenario: 0.0 for scenario in SCENARIOS}
        self.equity_paths = {scenario: [] for scenario in SCENARIOS}
        self.latest_close = {}
        self.counters = defaultdict(int)
        self.inventory_samples = []
        self.max_gross_leverage = 0.0
        self.max_gross_leverage_by_scenario = {
            scenario: 0.0 for scenario in SCENARIOS
        }
        self.margin_events = {"base": None, "adverse": None}
        self.base_guard_freeze_time = None
        self.source_hasher = hashlib.sha256()
        self.source_count = 0

    def _raw_mark_pips(self, position, when):
        bar, _ = self.market.completed(position.pair, when)
        if bar is None:
            return None
        return signed_pips(
            position.pair, position.side, position.entry_mid, bar.mid("c")
        )

    def _raw_equity(self, when):
        return self._scenario_equity(when, "raw")

    def _position_mark_jpy(self, position, when, scenario):
        # A just-entered position is marked at its causal entry open.  Older
        # inventory is marked only from a bar completed at this timestamp.
        if position.entry_time == when:
            bar = self.market.bar_open(position.pair, when)
            field = "o"
        else:
            bar, _ = self.market.completed(position.pair, when)
            field = "c"
        if bar is None:
            return None
        if scenario == "raw":
            pips = signed_pips(
                position.pair,
                position.side,
                position.entry_mid,
                bar.mid(field),
            )
        else:
            observed_exit = bar.executable(position.side, field)
            pips = signed_pips(
                position.pair,
                position.side,
                position.entry_observed,
                observed_exit,
            )
            slip = 0.3 if scenario == "base" else 0.9
            pips -= 2.0 * slip
        return self.market.pips_to_jpy(position.pair, pips, when, field)

    def _scenario_equity(self, when, scenario):
        equity = (
            self.prereg["portfolio"]["initial_equity_jpy"]
            + self.realized_jpy[scenario]
        )
        for position in self.open_positions.values():
            mark = self._position_mark_jpy(position, when, scenario)
            if mark is None:
                return None
            equity += mark
        return equity

    def _record_equity(self, when):
        gross_notional = self._gross_notional_jpy(when)
        if gross_notional is None:
            self.counters["gross_notional_mark_unavailable"] += 1
        for scenario in SCENARIOS:
            equity = self._scenario_equity(when, scenario)
            if equity is None:
                self.counters[f"{scenario}_equity_mark_unavailable"] += 1
                continue
            self.equity_paths[scenario].append((when, equity))
            leverage = (
                None
                if equity <= 0 or gross_notional is None
                else gross_notional / equity
            )
            if leverage is not None:
                self.max_gross_leverage_by_scenario[scenario] = max(
                    self.max_gross_leverage_by_scenario[scenario], leverage
                )
            if scenario not in self.margin_events:
                continue
            breached = equity <= 0 or (
                leverage is not None
                and leverage > self.prereg["portfolio"]["gross_leverage_cap"]
            )
            if breached and self.margin_events[scenario] is None:
                self.margin_events[scenario] = {
                    "time_utc": iso_utc(when),
                    "reason": "NONPOSITIVE_EQUITY"
                    if equity <= 0
                    else "GROSS_LEVERAGE_CAP",
                    "marked_equity_jpy": equity,
                    "marked_pnl_jpy": equity
                    - self.prereg["portfolio"]["initial_equity_jpy"],
                    "gross_notional_jpy": gross_notional,
                    "gross_leverage": leverage,
                    "marked_liquidation_value_recorded": True,
                    "changes_shared_policy_lineage": scenario == "base",
                }

    def _gross_notional_jpy(self, when):
        total = 0.0
        for position in self.open_positions.values():
            field = "o" if position.entry_time == when else "c"
            value = self.market.notional_jpy(position.pair, when, field)
            if value is None:
                return None
            total += value
        return total

    def _base_margin_guard(self, when):
        if not self.open_positions:
            return False
        equity = self._scenario_equity(when, "base")
        gross = self._gross_notional_jpy(when)
        leverage = (
            None if equity is None or equity <= 0 or gross is None else gross / equity
        )
        breached = equity is None or equity <= 0 or gross is None or (
            leverage is not None
            and leverage > self.prereg["portfolio"]["gross_leverage_cap"]
        )
        if not breached:
            return False
        for position in sorted(
            self.open_positions.values(), key=lambda value: value.trade_id
        ):
            self._request_exit(
                position,
                when,
                when + BAR_SECONDS,
                "BASE_MARGIN_HARD_GUARD",
            )
        self.base_guard_freeze_time = when
        self.counters["base_margin_hard_guard_events"] += 1
        self.counters["base_margin_hard_guard_lots"] += len(
            self.open_positions
        )
        return True

    def _reserved(self):
        reserved = [
            (position.pair, position.side) for position in self.open_positions.values()
        ]
        reserved.extend(
            (pending.signal.pair, pending.signal.side)
            for rows in self.pending_entries.values()
            for pending in rows
        )
        return reserved

    def _cap_allows(self, signal, when):
        reserved = self._reserved()
        pair_rows = [side for pair, side in reserved if pair == signal.pair]
        if len(pair_rows) >= self.prereg["portfolio"]["pair_lot_cap"]:
            return False, "PAIR_CAP"
        if any(side != signal.side for side in pair_rows):
            return False, "OPPOSITE_SAME_PAIR"
        if len(reserved) >= self.prereg["portfolio"]["gross_lot_cap"]:
            return False, "GROSS_CAP"
        usd_net = sum(usd_node_sign(pair, side) for pair, side in reserved)
        candidate_usd_sign = usd_node_sign(signal.pair, signal.side)
        usd_net += candidate_usd_sign
        if abs(usd_net) > self.prereg["portfolio"]["absolute_net_usd_node_lot_cap"]:
            return False, "USD_NODE_CAP"
        same_sign_count = sum(
            usd_node_sign(pair, side) == candidate_usd_sign
            for pair, side in reserved
        ) + 1
        if same_sign_count > self.prereg["portfolio"][
            "same_sign_usd_node_lot_cap"
        ]:
            return False, "USD_NODE_SAME_SIGN_CAP"
        # BASE marked equity is the one shared execution-safety ledger.  RAW
        # proposals remain cost-blind, and ADVERSE never rejects an entry.
        equity = self._scenario_equity(when, "base")
        if equity is None or equity <= 0:
            return False, "BASE_EQUITY_UNAVAILABLE_OR_NONPOSITIVE"
        notionals = []
        for pair, _ in reserved + [(signal.pair, signal.side)]:
            notional = self.market.notional_jpy(pair, when)
            if notional is None:
                return False, "NOTIONAL_UNAVAILABLE"
            notionals.append(notional)
        leverage = sum(notionals) / equity
        if leverage > self.prereg["portfolio"]["gross_leverage_cap"]:
            return False, "LEVERAGE_CAP"
        self.max_gross_leverage = max(self.max_gross_leverage, leverage)
        return True, None

    def _request_exit(self, position, action_time, exit_time, reason):
        if exit_time <= action_time:
            raise ValueError("dynamic exit must use a strictly later open")
        if exit_time < position.scheduled_exit_time:
            position.scheduled_exit_time = exit_time
            position.exit_action_time = action_time
            position.exit_reason = reason

    def _close(
        self,
        position,
        when,
        field,
        reason=None,
        action_time=None,
        price_time=None,
    ):
        price_time = when if price_time is None else price_time
        bar = (
            self.market.bar_open(position.pair, price_time)
            if field == "o"
            else self.market.completed(position.pair, price_time)[0]
        )
        if bar is None:
            raise ValueError("exit price unavailable")
        raw_exit = bar.mid(field)
        observed_exit = bar.executable(position.side, field)
        raw_pips = signed_pips(
            position.pair, position.side, position.entry_mid, raw_exit
        )
        observed_pips = signed_pips(
            position.pair, position.side, position.entry_observed, observed_exit
        )
        base_slip = 2.0 * 0.3
        adverse_slip = 2.0 * 0.9
        base_pips = observed_pips - base_slip
        adverse_pips = observed_pips - adverse_slip
        raw_jpy = self.market.pips_to_jpy(
            position.pair, raw_pips, price_time, field
        )
        base_jpy = self.market.pips_to_jpy(
            position.pair, base_pips, price_time, field
        )
        adverse_jpy = self.market.pips_to_jpy(
            position.pair, adverse_pips, price_time, field
        )
        terminal = (reason or position.exit_reason) in {
            "DATA_GAP_TERMINAL_MTM",
            "SPLIT_TERMINAL_LIQUIDATION",
        }
        trade = Trade(
            trade_id=position.trade_id,
            signal_id=position.signal_id,
            policy_id=position.policy_id,
            pair=position.pair,
            side=position.side,
            decision_time=position.decision_time,
            entry_time=position.entry_time,
            exit_action_time=(
                action_time if action_time is not None else position.exit_action_time
            ),
            exit_time=when,
            price_time=price_time,
            gap_detection_time=(
                when
                if (reason or position.exit_reason) == "DATA_GAP_TERMINAL_MTM"
                else None
            ),
            exit_reason=reason or position.exit_reason,
            holding_bars=max(
                0, (price_time - position.entry_time) // BAR_SECONDS
            ),
            raw_pips=raw_pips,
            base_pips=base_pips,
            adverse_pips=adverse_pips,
            raw_jpy=raw_jpy,
            base_jpy=base_jpy,
            adverse_jpy=adverse_jpy,
            raw_mfe_pips=position.raw_mfe_pips,
            raw_mae_pips=position.raw_mae_pips,
            terminal_liquidation=terminal,
        )
        self.trades.append(trade)
        for scenario, value in (
            ("raw", raw_jpy),
            ("base", base_jpy),
            ("adverse", adverse_jpy),
        ):
            if value is None:
                self.counters[f"{scenario}_pnl_conversion_missing"] += 1
            else:
                self.realized_jpy[scenario] += value
        self.open_positions.pop(position.trade_id, None)

    def _update_position(self, position, when, bar, feature):
        favorable = bar.mid("h") if position.side == "LONG" else bar.mid("l")
        adverse = bar.mid("l") if position.side == "LONG" else bar.mid("h")
        position.raw_mfe_pips = max(
            position.raw_mfe_pips,
            signed_pips(position.pair, position.side, position.entry_mid, favorable),
        )
        position.raw_mae_pips = min(
            position.raw_mae_pips,
            signed_pips(position.pair, position.side, position.entry_mid, adverse),
        )
        mark = signed_pips(
            position.pair, position.side, position.entry_mid, bar.mid("c")
        )
        position.peak_close_mfe_pips = max(position.peak_close_mfe_pips, mark)
        if mark >= position.tp_pips:
            position.tp_reached = True
        if feature is None:
            position.last_state = "UNKNOWN"
            position.last_state_time = when
            return mark
        pe_floor = self.calibration["path_efficiency_q67"][position.pair]["value"]
        spread_floor = self.calibration["spread_slot_q75"][position.pair][
            feature.slot
        ]["value"]
        momentum_opposed = (
            feature.impulse_side == opposite(position.side)
            and feature.path_efficiency >= pe_floor
        )
        momentum_aligned = feature.impulse_side == position.side
        rail_opposed = feature.rail_side == opposite(position.side)
        rail_aligned = feature.rail_side == position.side
        usd_position = usd_node_sign(position.pair, position.side)
        breadth_opposed = (
            feature.usd_breadth_count >= 2
            and feature.usd_breadth * usd_position < 0
        )
        adverse_votes = sum(
            int(value) for value in (momentum_opposed, rail_opposed, breadth_opposed)
        )
        trapped = feature.path_efficiency >= pe_floor and adverse_votes >= 2
        spread_good = feature.spread_pips <= spread_floor
        harvest = (
            mark > 0
            and spread_good
            and not trapped
            and not breadth_opposed
            and (momentum_aligned or rail_aligned)
        )
        age = max(0, (when - position.entry_time) // BAR_SECONDS)
        if trapped:
            state = "TRAPPED"
        elif harvest:
            state = "HARVEST"
        elif age >= 24:
            state = "STALE"
        else:
            state = "NEUTRAL"
        position.last_state = state
        position.last_state_time = when
        return mark

    def _basket_unwind(self, when, positions):
        groups = defaultdict(list)
        for position in positions:
            groups[usd_node_sign(position.pair, position.side)].append(position)
        selected = []
        for usd_sign, group in groups.items():
            if not any(position.last_state in {"STALE", "TRAPPED"} for position in group):
                continue
            if not all(position.last_state_time == when for position in group):
                continue
            total = 0.0
            valid = True
            for position in group:
                mark = self._raw_mark_pips(position, when)
                value = (
                    None
                    if mark is None
                    else self.market.pips_to_jpy(position.pair, mark, when, "c")
                )
                if value is None:
                    valid = False
                    break
                total += value
            if valid and total >= 0:
                selected.append((usd_sign, total, group))
        return selected

    def _dynamic_actions(self, when, positions):
        marks = {}
        for position in positions:
            # Every pre-existing position was updated exactly once at this
            # completed-data checkpoint by run().  Reuse that frozen state so
            # action logic cannot consume the same completed bar twice.
            if position.last_state_time != when:
                continue
            mark = self._raw_mark_pips(position, when)
            if mark is not None:
                marks[position.trade_id] = mark
        if self.policy_id == "P7":
            for _, _, group in self._basket_unwind(when, positions):
                for position in group:
                    self._request_exit(
                        position,
                        when,
                        when + BAR_SECONDS,
                        "USD_NODE_BASKET_UNWIND",
                    )
                    self.counters["basket_unwind_lots"] += 1
        for position in positions:
            if position.trade_id not in self.open_positions:
                continue
            if position.last_state_time != when:
                continue
            mark = marks.get(position.trade_id)
            if mark is None:
                continue
            if self.policy_id in {"P3", "P4"} and mark >= position.tp_pips:
                self._request_exit(
                    position, when, when + BAR_SECONDS, "RAW_CLOSE_MFE_Q40_TP"
                )
                continue
            if self.policy_id not in {"P5", "P6", "P7"}:
                continue
            if position.last_state == "TRAPPED":
                self._request_exit(
                    position, when, when + BAR_SECONDS, "LGAR_TRAPPED"
                )
                continue
            if self.policy_id in {"P6", "P7"} and position.tp_reached:
                if mark <= 0.5 * position.peak_close_mfe_pips:
                    self._request_exit(
                        position,
                        when,
                        when + BAR_SECONDS,
                        "CLOSE_MFE_50PCT_GIVEBACK",
                    )
                    continue
            age = max(0, (when - position.entry_time) // BAR_SECONDS)
            if age >= 24 and position.last_state != "HARVEST":
                self._request_exit(
                    position, when, when + BAR_SECONDS, "LGAR_NEUTRAL_24"
                )

    def _process_entries(self, when):
        pending = self.pending_entries.pop(when, [])
        for entry in pending:
            signal = entry.signal
            bar = self.market.bar_open(signal.pair, when)
            bars = self.market.bars_by_pair[signal.pair]
            source_index = signal.feature_index
            # This check happens only when the promised fill timestamp arrives.
            # It neither peeks from the decision timestamp nor permits a fill
            # that jumps over a missing decision-boundary bar.
            exact_path_arrived = (
                source_index + 2 < len(bars)
                and bars[source_index + 1].time == signal.decision_time
                and bars[source_index + 2].time == when
                and when == signal.decision_time + BAR_SECONDS
            )
            if bar is None or not exact_path_arrived:
                self.counters["entry_gap_unfilled"] += 1
                continue
            # Inventory may have changed between decision and fill because
            # precommitted exits execute first at this timestamp.  Recheck all
            # hard caps causally at the actual open; never allow a previously
            # reserved intent to breach a currency or gross cap.
            allowed, reason = self._cap_allows(signal, when)
            if not allowed:
                self.counters["fill_cap_skip_" + reason.lower()] += 1
                continue
            horizon = policy_horizon(self.policy_id)
            tp_pips = self.calibration["mfe_q40_pips"][signal.pair][signal.side][
                "value"
            ]
            trade_id = sha_bytes(
                canonical(
                    {
                        "signal_id": signal.signal_id,
                        "policy_id": self.policy_id,
                        "entry_time": iso_utc(when),
                    }
                )
            )
            self.open_positions[trade_id] = Position(
                trade_id=trade_id,
                policy_id=self.policy_id,
                signal_id=signal.signal_id,
                pair=signal.pair,
                side=signal.side,
                decision_time=signal.decision_time,
                entry_time=when,
                entry_mid=bar.mid("o"),
                entry_observed=bar.entry_executable(signal.side),
                fixed_exit_time=when + horizon * BAR_SECONDS,
                scheduled_exit_time=when + horizon * BAR_SECONDS,
                exit_action_time=when,
                exit_reason=f"TIME_EXIT_{horizon}",
                tp_pips=tp_pips,
            )
            self.counters["entries"] += 1

    def _process_due_exits(self, when):
        due = [
            position
            for position in self.open_positions.values()
            if position.scheduled_exit_time == when
        ]
        for position in sorted(due, key=lambda value: value.trade_id):
            if self.market.bar_open(position.pair, when) is None:
                continue
            self._close(position, when, "o")

    def _terminal_pair(self, pair, when, reason, detection_time=None):
        positions = [
            position
            for position in self.open_positions.values()
            if position.pair == pair
        ]
        for position in sorted(positions, key=lambda value: value.trade_id):
            event_time = when if detection_time is None else detection_time
            self._close(
                position,
                event_time,
                "c",
                reason=reason,
                action_time=event_time,
                price_time=when,
            )
            self.counters[reason.lower()] += 1
        for fill_time in list(self.pending_entries):
            retained = [
                entry
                for entry in self.pending_entries[fill_time]
                if entry.signal.pair != pair
            ]
            removed = len(self.pending_entries[fill_time]) - len(retained)
            self.counters["pending_entry_gap_cancelled"] += removed
            if retained:
                self.pending_entries[fill_time] = retained
            else:
                del self.pending_entries[fill_time]

    def _record_sources(self, when):
        signals = self.signals_by_time.get(when, [])
        for signal in signals:
            self.source_hasher.update(
                canonical(
                    {
                        "signal_id": signal.signal_id,
                        "pair": signal.pair,
                        "side": signal.side,
                        "decision_time": iso_utc(signal.decision_time),
                    }
                )
                + b"\n"
            )
            self.source_count += 1
            fill_time = when + BAR_SECONDS
            # Future feed availability is unknowable at decision time.  The
            # intent is reserved now and checked causally only when T+5 arrives.
            if self.base_guard_freeze_time == when:
                self.counters["cap_skip_base_margin_guard_freeze"] += 1
                continue
            allowed, reason = self._cap_allows(signal, when)
            if not allowed:
                self.counters["cap_skip_" + reason.lower()] += 1
                continue
            self.pending_entries[fill_time].append(PendingEntry(signal, fill_time))

    def _expire_missed_entries(self, when):
        # A timestamp absent from every pair produces no event at its promised
        # fill time.  Expire such intents at the first later observed event;
        # never carry them across the gap and never fill them late.
        for fill_time in sorted(
            value for value in self.pending_entries if value < when
        ):
            rows = self.pending_entries.pop(fill_time)
            self.counters["entry_gap_unfilled"] += len(rows)

    def run(self):
        event_times = {self.start, self.end}
        for pair, bars in self.market.bars_by_pair.items():
            for bar in bars:
                close = bar.time + BAR_SECONDS
                if self.start <= close <= self.end:
                    event_times.add(close)
                if self.start <= bar.time <= self.end:
                    event_times.add(bar.time)
        for when in sorted(event_times):
            self._expire_missed_entries(when)
            # Detect an already-missed close only when its timestamp has
            # arrived (or at the first later event if every pair was silent).
            # Accounting uses the last observed executable close, but the
            # detection/action timestamp is never backdated to that mark.
            for pair in list(self.market.bars_by_pair):
                if not any(
                    position.pair == pair
                    for position in self.open_positions.values()
                ):
                    continue
                latest = self.latest_close.get(pair)
                if latest is None:
                    continue
                expected_close = latest[0] + BAR_SECONDS
                if (
                    when >= expected_close
                    and self.market.completed(pair, expected_close)[0] is None
                ):
                    self._terminal_pair(
                        pair,
                        latest[0],
                        "DATA_GAP_TERMINAL_MTM",
                        detection_time=when,
                    )
            for pair in self.market.bars_by_pair:
                bar, _ = self.market.completed(pair, when)
                if bar is not None:
                    self.latest_close[pair] = (when, bar.mid("c"))
            preexisting = list(self.open_positions.values())
            for position in preexisting:
                bar, feature = self.market.completed(position.pair, when)
                if bar is not None:
                    self._update_position(position, when, bar, feature)
            if when == self.end:
                for pair in list(self.market.bars_by_pair):
                    if any(
                        position.pair == pair
                        for position in self.open_positions.values()
                    ):
                        if self.market.completed(pair, when)[0] is None:
                            raise ValueError("split terminal close unavailable")
                        self._terminal_pair(
                            pair, when, "SPLIT_TERMINAL_LIQUIDATION"
                        )
                self.pending_entries.clear()
                self._record_equity(when)
                break
            self._process_due_exits(when)
            self._process_entries(when)
            self._base_margin_guard(when)
            still_preexisting = [
                position
                for position in preexisting
                if position.trade_id in self.open_positions
            ]
            self._dynamic_actions(when, still_preexisting)
            self._record_sources(when)
            self._record_equity(when)
            gross = len(self.open_positions) + sum(
                len(rows) for rows in self.pending_entries.values()
            )
            usd_node = abs(
                sum(
                    usd_node_sign(pair, side)
                    for pair, side in self._reserved()
                )
            )
            pair_counts = defaultdict(int)
            usd_sign_counts = defaultdict(int)
            for pair, _ in self._reserved():
                pair_counts[pair] += 1
            for pair, side in self._reserved():
                usd_sign_counts[usd_node_sign(pair, side)] += 1
            max_pair_lots = max(pair_counts.values(), default=0)
            max_same_sign_usd_lots = max(usd_sign_counts.values(), default=0)
            self.inventory_samples.append(gross)
            self.counters["max_open_inventory"] = max(
                self.counters["max_open_inventory"], len(self.open_positions)
            )
            self.counters["max_reserved_inventory"] = max(
                self.counters["max_reserved_inventory"], gross
            )
            self.counters["max_abs_usd_node_lots"] = max(
                self.counters["max_abs_usd_node_lots"], usd_node
            )
            self.counters["max_pair_lots"] = max(
                self.counters["max_pair_lots"], max_pair_lots
            )
            self.counters["max_same_sign_usd_node_lots"] = max(
                self.counters["max_same_sign_usd_node_lots"],
                max_same_sign_usd_lots,
            )
        if self.open_positions or self.pending_entries:
            raise AssertionError("simulation ended with inventory or pending entries")
        if self.counters.get("max_reserved_inventory", 0) > self.prereg[
            "portfolio"
        ]["gross_lot_cap"]:
            raise AssertionError("gross lot cap breached")
        if self.counters.get("max_abs_usd_node_lots", 0) > self.prereg[
            "portfolio"
        ]["absolute_net_usd_node_lot_cap"]:
            raise AssertionError("USD-node cap breached")
        if self.counters.get("max_same_sign_usd_node_lots", 0) > self.prereg[
            "portfolio"
        ]["same_sign_usd_node_lot_cap"]:
            raise AssertionError("same-sign USD-node cap breached")
        if self.counters.get("max_pair_lots", 0) > self.prereg["portfolio"][
            "pair_lot_cap"
        ]:
            raise AssertionError("pair lot cap breached")
        return {
            "policy_id": self.policy_id,
            "split": self.split_name,
            "source_signal_count": self.source_count,
            "source_signal_sha256": self.source_hasher.hexdigest(),
            "trades": self.trades,
            "counters": dict(sorted(self.counters.items())),
            "inventory_samples": self.inventory_samples,
            "max_gross_leverage": self.max_gross_leverage,
            "max_gross_leverage_by_scenario": self.max_gross_leverage_by_scenario,
            "margin_events": self.margin_events,
            "equity_paths": self.equity_paths,
        }


def _scenario_value(trade, scenario, suffix="pips"):
    return getattr(trade, f"{scenario}_{suffix}")


def cluster_bootstrap_lcb(trades, scenario, alpha, resamples, seed_text):
    rows = [
        (utc_day(trade.decision_time), _scenario_value(trade, scenario))
        for trade in trades
    ]
    return cluster_bootstrap_value_lcb(
        rows, alpha, resamples, seed_text
    )


def cluster_bootstrap_value_lcb(rows, alpha, resamples, seed_text):
    """Lower quantile after resampling whole UTC-decision-day clusters."""
    clusters = defaultdict(lambda: [0.0, 0])
    for day, value in rows:
        clusters[day][0] += value
        clusters[day][1] += 1
    keys = sorted(clusters)
    if len(keys) < 2:
        return None
    rng = random.Random(int(sha_bytes(seed_text.encode("utf-8"))[:16], 16))
    samples = []
    for _ in range(resamples):
        total = 0.0
        count = 0
        for _ in keys:
            selected = keys[rng.randrange(len(keys))]
            subtotal, subcount = clusters[selected]
            total += subtotal
            count += subcount
        samples.append(total / count)
    return quantile(samples, alpha)


def _equity_metrics(trades, equity_path, scenario, initial_equity):
    if not equity_path:
        raise ValueError("equity path is required for MTM metrics")
    peak = float(initial_equity)
    max_drawdown = 0.0
    month_open = {}
    month_close = {}
    nonpositive_months = set()
    first_nonpositive = None
    prior_equity = float(initial_equity)
    prior_time = None
    for when, equity in equity_path:
        if prior_time is not None and when <= prior_time:
            raise ValueError("equity path timestamps must strictly increase")
        month = utc_month(when)
        month_open.setdefault(month, prior_equity)
        month_close[month] = equity
        if equity <= 0:
            nonpositive_months.add(month)
            if first_nonpositive is None:
                first_nonpositive = when
        peak = max(peak, equity)
        if peak > 0:
            max_drawdown = min(max_drawdown, equity / peak - 1.0)
        prior_equity = equity
        prior_time = when
    multiples = {
        month: (
            month_close[month] / month_open[month]
            if month_open[month] > 0 and month not in nonpositive_months
            else None
        )
        for month in month_open
    }
    valid = [value for value in multiples.values() if value is not None]
    missing = sum(
        _scenario_value(trade, scenario, "jpy") is None for trade in trades
    )
    end_equity = equity_path[-1][1]
    return {
        "end_equity_jpy": end_equity,
        "equity_multiple": end_equity / initial_equity,
        "minimum_mtm_equity_jpy": min(value for _, value in equity_path),
        "max_drawdown_fraction": max_drawdown,
        "pnl_conversion_missing": missing,
        "equity_path_rows": len(equity_path),
        "first_nonpositive_equity_time_utc": (
            iso_utc(first_nonpositive) if first_nonpositive is not None else None
        ),
        "ruin_observed": first_nonpositive is not None,
        "monthly_multiples_suppressed_for_nonpositive_equity": sorted(
            nonpositive_months
        ),
        "monthly_multiples": multiples,
        "positive_month_fraction": (
            sum(value > 1.0 for value in valid) / len(valid) if valid else 0.0
        ),
        "worst_month_multiple": min(valid) if valid else None,
        "best_month_multiple": max(valid) if valid else None,
    }


def summarize_simulation(simulation, prereg):
    trades = simulation["trades"]
    split = prereg["splits"][simulation["split"]]
    duration_days = (
        parse_time(split["to_utc"]) - parse_time(split["from_utc"])
    ) / 86400.0
    family_alpha = prereg["selection"]["alpha"] / prereg["selection"][
        "family_size"
    ]
    resamples = prereg["selection"]["cluster_bootstrap_resamples"]
    scenario_metrics = {}
    for scenario in SCENARIOS:
        pips = [_scenario_value(trade, scenario) for trade in trades]
        wins = [value for value in pips if value > 0]
        losses = [value for value in pips if value < 0]
        scenario_metrics[scenario] = {
            "mean_pips_per_trade": statistics.fmean(pips) if pips else None,
            "total_pips": sum(pips),
            "direction_accuracy": sum(value > 0 for value in pips) / len(pips)
            if pips
            else None,
            "profit_factor_pips": (
                sum(wins) / abs(sum(losses)) if losses else None
            ),
            "bonferroni_day_cluster_bootstrap_lcb_pips": cluster_bootstrap_lcb(
                trades,
                scenario,
                family_alpha,
                resamples,
                f"{simulation['split']}|{simulation['policy_id']}|{scenario}",
            ),
            **_equity_metrics(
                trades,
                simulation["equity_paths"][scenario],
                scenario,
                prereg["portfolio"]["initial_equity_jpy"],
            ),
        }
    pair_metrics = {}
    for pair in prereg["inputs"]["symbols"]:
        rows = [trade for trade in trades if trade.pair == pair]
        pair_metrics[pair] = {
            "trades": len(rows),
            "raw_mean_pips": statistics.fmean(
                [trade.raw_pips for trade in rows]
            )
            if rows
            else None,
            "base_mean_pips": statistics.fmean(
                [trade.base_pips for trade in rows]
            )
            if rows
            else None,
            "adverse_mean_pips": statistics.fmean(
                [trade.adverse_pips for trade in rows]
            )
            if rows
            else None,
        }
    ages = [trade.holding_bars for trade in trades]
    mfe_values = [trade.raw_mfe_pips for trade in trades]
    mae_values = [trade.raw_mae_pips for trade in trades]
    inventory = simulation["inventory_samples"]
    reasons = defaultdict(int)
    for trade in trades:
        reasons[trade.exit_reason] += 1
    split_terminal = [
        trade
        for trade in trades
        if trade.exit_reason == "SPLIT_TERMINAL_LIQUIDATION"
    ]
    gap_terminal = [
        trade
        for trade in trades
        if trade.exit_reason == "DATA_GAP_TERMINAL_MTM"
    ]

    def terminal_values(rows):
        return {
            scenario: sum(
                value
                for value in (
                    _scenario_value(trade, scenario, "jpy") for trade in rows
                )
                if value is not None
            )
            for scenario in SCENARIOS
        }
    lineage_material = [
        {
            "signal_id": trade.signal_id,
            "policy_action_time": iso_utc(trade.exit_action_time),
            "exit_time": iso_utc(trade.exit_time),
            "price_time": iso_utc(trade.price_time),
            "gap_detection_time": (
                iso_utc(trade.gap_detection_time)
                if trade.gap_detection_time is not None
                else None
            ),
            "exit_reason": trade.exit_reason,
        }
        for trade in sorted(trades, key=lambda value: value.trade_id)
    ]
    lineage_sha = sha_bytes(canonical(lineage_material))
    base_mean = scenario_metrics["base"]["mean_pips_per_trade"]
    raw_mean = scenario_metrics["raw"]["mean_pips_per_trade"]
    return {
        "policy_id": simulation["policy_id"],
        "source_signal_count": simulation["source_signal_count"],
        "source_signal_sha256": simulation["source_signal_sha256"],
        "raw_proposals_per_calendar_day": simulation["source_signal_count"]
        / duration_days,
        "executed_trades": len(trades),
        "executed_trades_per_calendar_day": len(trades) / duration_days,
        "utc_decision_day_clusters": len(
            {utc_day(trade.decision_time) for trade in trades}
        ),
        "scenario_metrics": scenario_metrics,
        "gross_to_base_cost_drag_pips": (
            raw_mean - base_mean
            if raw_mean is not None and base_mean is not None
            else None
        ),
        "break_even_round_trip_cost_pips": raw_mean,
        "pair_metrics": pair_metrics,
        "positive_pair_count_raw": sum(
            row["raw_mean_pips"] is not None and row["raw_mean_pips"] > 0
            for row in pair_metrics.values()
        ),
        "holding_age_bars": {
            "q50": quantile(ages, 0.50) if ages else None,
            "q90": quantile(ages, 0.90) if ages else None,
            "q99": quantile(ages, 0.99) if ages else None,
            "max": max(ages) if ages else None,
        },
        "post_entry_excursion_pips": {
            "raw_mfe_mean": statistics.fmean(mfe_values)
            if mfe_values
            else None,
            "raw_mfe_q50": quantile(mfe_values, 0.50)
            if mfe_values
            else None,
            "raw_mfe_q90": quantile(mfe_values, 0.90)
            if mfe_values
            else None,
            "raw_mae_mean": statistics.fmean(mae_values)
            if mae_values
            else None,
            "raw_mae_q10": quantile(mae_values, 0.10)
            if mae_values
            else None,
            "raw_mae_q50": quantile(mae_values, 0.50)
            if mae_values
            else None,
        },
        "inventory": {
            "sample_q50": quantile(inventory, 0.50) if inventory else 0,
            "sample_q90": quantile(inventory, 0.90) if inventory else 0,
            "sample_q99": quantile(inventory, 0.99) if inventory else 0,
            "terminal_open_inventory": 0,
            "terminal_open_inventory_mtm_jpy": 0.0,
            "split_terminal_liquidation_lots": len(split_terminal),
            "split_terminal_liquidation_pnl_jpy": terminal_values(
                split_terminal
            ),
            "data_gap_terminal_mtm_lots": len(gap_terminal),
            "data_gap_terminal_mtm_pnl_jpy": terminal_values(gap_terminal),
            "terminal_liquidation_included_in_equity": True,
            "max_gross_leverage": simulation["max_gross_leverage"],
            "max_gross_leverage_by_scenario": simulation[
                "max_gross_leverage_by_scenario"
            ],
            "max_abs_usd_node_lots": simulation["counters"].get(
                "max_abs_usd_node_lots", 0
            ),
            "max_pair_lots": simulation["counters"].get(
                "max_pair_lots", 0
            ),
            "max_same_sign_usd_node_lots": simulation["counters"].get(
                "max_same_sign_usd_node_lots", 0
            ),
            "margin_events": simulation["margin_events"],
        },
        "exit_reasons": dict(sorted(reasons.items())),
        "counters": simulation["counters"],
        "arm_lineage_sha256": {
            "RAW_SIGNAL": lineage_sha,
            "EXECUTABLE_BASE": lineage_sha,
            "ADVERSE_STRESS": lineage_sha,
        },
        "arm_lineage_identical": True,
    }


def _daily_mtm_pnl(equity_path, initial_equity):
    day_closes = {}
    for when, equity in equity_path:
        day_closes[utc_day(when)] = equity
    result = {}
    prior = float(initial_equity)
    for day in sorted(day_closes):
        result[day] = day_closes[day] - prior
        prior = day_closes[day]
    return result


def paired_improvement(candidate, baseline, prereg):
    if (
        candidate["source_signal_count"] != baseline["source_signal_count"]
        or candidate["source_signal_sha256"] != baseline["source_signal_sha256"]
    ):
        raise AssertionError("paired policies must share the complete RAW proposal stream")
    source_count = candidate["source_signal_count"]
    initial = prereg["portfolio"]["initial_equity_jpy"]
    completeness = {}
    for label, simulation in (("candidate", candidate), ("baseline", baseline)):
        missing_trade_values = sum(
            any(
                _scenario_value(trade, scenario, "jpy") is None
                for scenario in SCENARIOS
            )
            for trade in simulation["trades"]
        )
        missing_marks = sum(
            simulation["counters"].get(
                f"{scenario}_equity_mark_unavailable", 0
            )
            for scenario in SCENARIOS
        )
        missing_notional = simulation["counters"].get(
            "gross_notional_mark_unavailable", 0
        )
        completeness[label] = {
            "missing_trade_conversion_rows": missing_trade_values,
            "missing_equity_mark_rows": missing_marks,
            "missing_gross_notional_rows": missing_notional,
            "complete": (
                missing_trade_values == 0
                and missing_marks == 0
                and missing_notional == 0
            ),
        }
    valid = all(row["complete"] for row in completeness.values())
    result = {
        "baseline_policy_id": "P2",
        "shared_source_signal_count": source_count,
        "shared_source_signal_sha256": candidate["source_signal_sha256"],
        "comparison_unit": "full daily MTM portfolio PnL; no executed-signal intersection",
        "data_completeness": completeness,
        "valid_for_evidence": valid,
        "invalid_reason": None
        if valid
        else "UNPRICEABLE_OR_MISSING_CONVERSION_ROWS",
    }
    for scenario in SCENARIOS:
        candidate_days = _daily_mtm_pnl(candidate["equity_paths"][scenario], initial)
        baseline_days = _daily_mtm_pnl(baseline["equity_paths"][scenario], initial)
        days = sorted(set(candidate_days) | set(baseline_days))
        deltas = [candidate_days.get(day, 0.0) - baseline_days.get(day, 0.0) for day in days]
        candidate_end = candidate["equity_paths"][scenario][-1][1]
        baseline_end = baseline["equity_paths"][scenario][-1][1]
        terminal_delta = candidate_end - baseline_end
        result[scenario] = {
            "paired_utc_days": len(days),
            "terminal_equity_delta_jpy": terminal_delta,
            "total_pnl_delta_jpy": terminal_delta,
            "total_pnl_delta_jpy_per_raw_proposal": (
                terminal_delta / source_count if source_count else None
            ),
            "mean_daily_mtm_pnl_delta_jpy": statistics.fmean(deltas)
            if deltas
            else None,
            "positive_delta_fraction": sum(value > 0 for value in deltas) / len(deltas)
            if deltas
            else None,
            "day_bootstrap_lcb_mean_daily_delta_jpy": cluster_bootstrap_value_lcb(
                list(zip(days, deltas)),
                prereg["selection"]["alpha"],
                prereg["selection"]["cluster_bootstrap_resamples"],
                f"paired|{candidate['split']}|{candidate['policy_id']}|P2|{scenario}",
            ),
        }
    return result


def _metric(summary, scenario, name):
    return summary["scenario_metrics"][scenario][name]


def admission_checks(tuning, prereg):
    density = prereg["selection"]["density_floor"]
    stability = prereg["selection"]["stability_floor"]
    checks = {
        "tuning_density_executed_trades": tuning["executed_trades"]
        >= density["executed_trades"],
        "tuning_density_decision_days": tuning["utc_decision_day_clusters"]
        >= density["utc_decision_days"],
        "tuning_density_pairs": sum(
            row["trades"] >= 30 for row in tuning["pair_metrics"].values()
        )
        >= density["pairs_with_30_trades"],
        "tuning_raw_mean_positive": _metric(
            tuning, "raw", "mean_pips_per_trade"
        )
        > 0,
        "tuning_raw_corrected_lcb_positive": _metric(
            tuning, "raw", "bonferroni_day_cluster_bootstrap_lcb_pips"
        )
        is not None
        and _metric(
            tuning, "raw", "bonferroni_day_cluster_bootstrap_lcb_pips"
        )
        > 0,
        "tuning_positive_pairs": tuning["positive_pair_count_raw"]
        >= stability["positive_pairs"],
        "tuning_positive_month_fraction": _metric(
            tuning, "raw", "positive_month_fraction"
        )
        >= stability["positive_month_fraction"],
        "terminal_inventory_zero": tuning["inventory"][
            "terminal_open_inventory_mtm_jpy"
        ]
        == 0.0,
        "tuning_inventory_caps_respected": tuning["inventory"][
            "max_abs_usd_node_lots"
        ]
        <= prereg["portfolio"]["absolute_net_usd_node_lot_cap"]
        and tuning["inventory"]["max_same_sign_usd_node_lots"]
        <= prereg["portfolio"]["same_sign_usd_node_lot_cap"]
        and tuning["inventory"]["max_pair_lots"]
        <= prereg["portfolio"]["pair_lot_cap"]
        and tuning["counters"].get("max_reserved_inventory", 0)
        <= prereg["portfolio"]["gross_lot_cap"],
        "tuning_base_margin_closeout_absent": tuning["inventory"][
            "margin_events"
        ]["base"]
        is None,
        "tuning_adverse_margin_closeout_absent": tuning["inventory"][
            "margin_events"
        ]["adverse"]
        is None,
        "tuning_base_ruin_absent": _metric(tuning, "base", "ruin_observed")
        is False,
        "tuning_adverse_ruin_absent": _metric(
            tuning, "adverse", "ruin_observed"
        )
        is False,
        "tuning_data_completeness": all(
            _metric(tuning, scenario, "pnl_conversion_missing") == 0
            for scenario in SCENARIOS
        )
        and tuning["counters"].get("gross_notional_mark_unavailable", 0) == 0
        and all(
            tuning["counters"].get(
                f"{scenario}_equity_mark_unavailable", 0
            )
            == 0
            for scenario in SCENARIOS
        ),
        "arm_lineage_identical": tuning["arm_lineage_identical"] is True,
    }
    return checks


def result_status(checks):
    ordered = [
        (
            "tuning_raw_corrected_lcb_positive",
            "REJECTED_TUNING_GROSS_EDGE_NOT_FAMILY_CORRECTED",
        ),
        ("tuning_raw_mean_positive", "REJECTED_NO_TUNING_GROSS_EDGE"),
        (
            "tuning_positive_month_fraction",
            "REJECTED_PERIOD_INSTABILITY",
        ),
        ("tuning_positive_pairs", "REJECTED_PAIR_INSTABILITY"),
        (
            "tuning_base_margin_closeout_absent",
            "REJECTED_BASE_MARGIN_CLOSEOUT",
        ),
        (
            "tuning_adverse_margin_closeout_absent",
            "REJECTED_ADVERSE_MARGIN_CLOSEOUT",
        ),
        ("tuning_data_completeness", "REJECTED_DATA_COMPLETENESS"),
    ]
    for key, status in ordered:
        if not checks.get(key, False):
            return status
    if not all(checks.values()):
        return "REJECTED_CONTRACT_OR_DENSITY_FAILURE"
    return "TUNING_PASS_REQUIRES_NEW_PREREG_FRESH_EVIDENCE_WINDOW"


def _safe_summary_for_result(summary):
    return summary


def run_replay():
    prereg = load_preregistration()
    contract_checks = validate_preregistration(prereg)
    prereg_sha = sha_file(PREREG_PATH)
    bars_by_pair, input_audit = load_inputs(prereg)
    features_by_pair = {
        pair: compute_features(pair, bars, prereg)
        for pair, bars in bars_by_pair.items()
    }
    attach_usd_breadth(features_by_pair)
    signals_by_time = make_signals(bars_by_pair, features_by_pair, prereg_sha)
    calibration = derive_calibration(
        bars_by_pair, features_by_pair, signals_by_time, prereg
    )
    market = Market(bars_by_pair, features_by_pair)

    tuning_runs = {}
    tuning_summaries = {}
    for policy_id in prereg["policies"]:
        simulation = PolicySimulator(
            policy_id,
            market,
            signals_by_time,
            calibration,
            prereg,
            "tuning",
        ).run()
        tuning_runs[policy_id] = simulation
        tuning_summaries[policy_id] = summarize_simulation(simulation, prereg)
    source_hashes = {
        summary["source_signal_sha256"] for summary in tuning_summaries.values()
    }
    source_counts = {
        summary["source_signal_count"] for summary in tuning_summaries.values()
    }
    if len(source_hashes) != 1 or len(source_counts) != 1:
        raise AssertionError("policies did not receive the identical proposal stream")
    selected_policy = max(
        tuning_summaries,
        key=lambda policy_id: (
            _metric(tuning_summaries[policy_id], "raw", "mean_pips_per_trade"),
            -int(policy_id[1:]),
        ),
    )

    selected_tuning = tuning_summaries[selected_policy]
    checks = admission_checks(selected_tuning, prereg)
    status = result_status(checks)
    paired_tuning_diagnostic = paired_improvement(
        tuning_runs[selected_policy], tuning_runs["P2"], prereg
    )
    body = {
        "schema": "QR_OANDA_POST_ENTRY_LGAR_RESULT_V1",
        "candidate_id": prereg["candidate_id"],
        "status": status,
        "strategy_admitted": False,
        "profitability_proven": False,
        "monthly_2x_proven": False,
        "live_order_authority": False,
        "broker_mutation_allowed": False,
        "external_order_attempts": 0,
        "external_orders": 0,
        "network_attempts": 0,
        "credential_reads": 0,
        "invalidated_pre_fix_run_receipt_sha256": sha_file(
            ROOT / "INVALIDATED_RUN_RECEIPT.json"
        ),
        "preregistration_sha256": prereg_sha,
        "contract_checks": contract_checks,
        "hard_guard_contract": {
            "shared_execution_equity_ledger": "EXECUTABLE_BASE",
            "raw_proposals_cost_blind": True,
            "adverse_stress_can_reject_entry": False,
            "adverse_margin_outcome_changes_shared_lineage": False,
            "base_or_adverse_margin_event_is_admission_failure": True,
        },
        "input_audit": input_audit,
        "opened_development": {
            "status": "INVALIDATED_DIAGNOSTIC_NOT_EVIDENCE",
            "final_run_price_or_volume_rows_decoded": 0,
            "final_run_labels_computed": 0,
            "prior_invalidated_result_receipt_sha256": sha_file(
                ROOT / "INVALIDATED_RUN_RECEIPT.json"
            ),
            "reuse_for_admission": False,
            "next_step_if_tuning_passes": "NEW_PREREGISTERED_VERSION_AND_FRESH_EVIDENCE_WINDOW",
        },
        "holdout": {
            "status": "UNTOUCHED_LOCKED",
            "from_utc": prereg["splits"]["untouched_holdout"]["from_utc"],
            "to_utc": prereg["splits"]["untouched_holdout"]["to_utc"],
            "price_or_volume_rows_decoded": 0,
            "labels_computed": 0,
        },
        "calibration": calibration,
        "tuning": {
            "selection_basis": "RAW_SIGNAL_ONLY",
            "family_size": 8,
            "source_signal_count": next(iter(source_counts)),
            "source_signal_sha256": next(iter(source_hashes)),
            "policies": {
                policy_id: _safe_summary_for_result(summary)
                for policy_id, summary in tuning_summaries.items()
            },
            "selected_policy_id": selected_policy,
            "selected_vs_time48_paired_diagnostic": paired_tuning_diagnostic,
        },
        "admission_checks": checks,
        "tuning_gate_passed": status
        == "TUNING_PASS_REQUIRES_NEW_PREREG_FRESH_EVIDENCE_WINDOW",
        "limitations": [
            "Untouched holdout was not read and is mandatory for final evidence.",
            "Opened development was exposed by invalidated implementation runs and is not decoded or reused by this remediation run.",
            "Zero financing is an explicit four-hour-vehicle approximation.",
            "Fixed-unit replay uses the frozen leverage-20 guard and does not model provider-specific margin tiers or closeout percentages.",
            "BASE uses the preregistered common leverage guard; ADVERSE margin closeout is a marked counterfactual and does not alter shared lineage.",
            "OANDA price-count volume is not traded volume and is unused.",
            "No profitability guarantee or live-order permission follows from this replay.",
        ],
    }
    result = dict(body)
    result["result_sha256"] = sha_bytes(canonical(body))
    atomic_json(RESULT_PATH, result)
    files = {
        "PREREGISTRATION.md": sha_file(ROOT / "PREREGISTRATION.md"),
        "PREREGISTRATION.json": sha_file(PREREG_PATH),
        "replay_lgar.py": sha_file(ROOT / "replay_lgar.py"),
        "test_replay_lgar.py": sha_file(ROOT / "test_replay_lgar.py"),
        "INVALIDATED_RUN_RECEIPT.json": sha_file(
            ROOT / "INVALIDATED_RUN_RECEIPT.json"
        ),
        "result.json": sha_file(RESULT_PATH),
    }
    packet_body = {
        "schema": "QR_OANDA_POST_ENTRY_LGAR_EVIDENCE_PACKET_V1",
        "candidate_id": prereg["candidate_id"],
        "status": status,
        "result_sha256": result["result_sha256"],
        "files": files,
        "canonical_dataset_sha256": prereg["inputs"][
            "canonical_dataset_sha256"
        ],
        "holdout_status": "UNTOUCHED_LOCKED",
        "opened_development_status": "INVALIDATED_DIAGNOSTIC_NOT_EVIDENCE",
        "opened_development_price_or_volume_rows_decoded": 0,
        "opened_development_labels_computed": 0,
        "holdout_price_or_volume_rows_decoded": 0,
        "holdout_labels_computed": 0,
        "network_attempts": 0,
        "credential_reads": 0,
        "external_order_attempts": 0,
        "external_orders": 0,
        "live_order_authority": False,
        "broker_mutation_allowed": False,
    }
    packet = dict(packet_body)
    packet["packet_sha256"] = sha_bytes(canonical(packet_body))
    atomic_json(PACKET_PATH, packet)
    return result, packet


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run",), nargs="?", default="run")
    args = parser.parse_args(argv)
    if args.command == "run":
        result, packet = run_replay()
        print(
            json.dumps(
                {
                    "candidate_id": result["candidate_id"],
                    "status": result["status"],
                    "selected_policy_id": result["tuning"]["selected_policy_id"],
                    "tuning_raw_mean_pips": result["tuning"]["policies"][
                        result["tuning"]["selected_policy_id"]
                    ]["scenario_metrics"]["raw"]["mean_pips_per_trade"],
                    "tuning_raw_corrected_lcb_pips": result["tuning"][
                        "policies"
                    ][result["tuning"]["selected_policy_id"]][
                        "scenario_metrics"
                    ]["raw"]["bonferroni_day_cluster_bootstrap_lcb_pips"],
                    "opened_development_rows_decoded": 0,
                    "holdout_rows_decoded": 0,
                    "result_sha256": result["result_sha256"],
                    "packet_sha256": packet["packet_sha256"],
                    "external_orders": 0,
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
