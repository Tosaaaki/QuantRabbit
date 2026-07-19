#!/usr/bin/env python3
"""Run a blind, preregistered DOJO capital-recycling TRAIN experiment.

``prepare`` selects six outcome-free cases and seals only cutoff-safe packets.
``seal-responses`` validates and first-write seals six capability-isolated model
judgments without reading future prices.  ``score`` is the only mode that
constructs market truth.  This program is historical research only: it has no
broker client, order method, live permission, or promotion authority.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
from bisect import bisect_left
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.analysis.market_status import compute_market_status


UTC = timezone.utc
PAIRS = (
    "EUR_USD",
    "USD_JPY",
    "GBP_USD",
    "AUD_USD",
    "USD_CHF",
    "USD_CAD",
    "EUR_JPY",
    "GBP_JPY",
)
# This exact six-month interval is deliberately later than the V1 experiment's
# 2025-12-01 upper bound.  It is already-worn history and can produce only a
# TRAIN diagnostic, never holdout or forward evidence.
FROM = datetime(2026, 1, 1, tzinfo=UTC)
TO = datetime(2026, 7, 1, tzinfo=UTC)
SEED = 2026071902
CELL_COUNT = 6
H1_BARS = 120
M5_BARS = 36
BREAK_BARS = 3
TREND_LOOKBACK_S = 24 * 60 * 60
TP_PIPS = 3.0
DECISION_DELAY_S = 60 * 60
HOLD_HORIZON_S = 4 * 60 * 60
MIN_CASE_SEPARATION_S = 24 * 60 * 60
EXPERIMENT_ID = "codex-ai-capital-recycle-train-v2"

PROMPT_TEMPLATE = """You are making exactly ONE blind DOJO direction judgment. No tools, files,
web, memory, other cases, or outside context may be used. The packet contains
only observations complete before this decision-bar open. ASSET_A is an
existing mechanical position held for about one hour. ASSET_B is a new
mechanical candidate triggered for this same open.

First decide whether ASSET_A's stated direction remains valid: return
KEEP_LONG/KEEP_SHORT matching its stated side, or FLAT. Independently decide
whether ASSET_B's stated direction is valid: return LONG/SHORT matching its
stated side, or FLAT. Do not choose percentages, leverage, lot size, or a
reverse direction. A fixed evaluator—not you—will apply this preregistered
capital policy: KEEP means 100% ASSET_A/full HOLD and ignores ASSET_B for
allocation; FLAT means 0% ASSET_A, then 100% ASSET_B only when its direction is
judged valid, otherwise 100% reserve.

Return JSON only:
{"id":"Pxx","existing_direction":"KEEP_LONG|KEEP_SHORT|FLAT",
"next_direction":"LONG|SHORT|FLAT","conviction":1,
"reason":"max 240 chars"}
conviction is an integer 1..3. Do not discuss any other case.
"""


def canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def raw_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    first_write_bytes(path, payload)


def first_write_bytes(path: Path, payload: bytes) -> None:
    """Publish bytes once with an OS-enforced create-new boundary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise ValueError(f"refusing to overwrite {path.name}") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # Only the writer that won O_EXCL can reach here. Remove its incomplete
        # inode so a later run cannot mistake a partial response for a seal.
        path.unlink(missing_ok=True)
        raise


def atomic_text(path: Path, value: str) -> None:
    first_write_bytes(path, value.encode("utf-8"))


def _body_with_verified_canonical_sha(
    document: dict[str, Any], label: str
) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise ValueError(f"{label} must be an object")
    body = dict(document)
    claimed = body.pop("canonical_sha256", None)
    if not isinstance(claimed, str) or canonical_sha(body) != claimed:
        raise ValueError(f"{label} canonical SHA-256 mismatch")
    return body


def pip_size(pair: str) -> float:
    return 0.01 if pair.endswith("JPY") else 0.0001


def load_data(root: Path) -> dict[str, list[tuple[Any, ...]]]:
    result: dict[str, list[tuple[Any, ...]]] = {}
    for pair in PAIRS:
        rows: list[tuple[Any, ...]] = []
        for shard in sorted(root.glob(f"*/{pair}/{pair}_M5_BA_*.jsonl.gz")):
            with gzip.open(shard, "rt", encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    epoch = int(
                        datetime.fromisoformat(row["time"][:19] + "+00:00").timestamp()
                    )
                    rows.append(
                        (
                            epoch,
                            float(row["bid"]["o"]),
                            float(row["bid"]["h"]),
                            float(row["bid"]["l"]),
                            float(row["bid"]["c"]),
                            float(row["ask"]["o"]),
                            float(row["ask"]["h"]),
                            float(row["ask"]["l"]),
                            float(row["ask"]["c"]),
                        )
                    )
        rows.sort()
        if not rows:
            raise ValueError(f"no rows for {pair}")
        if any(rows[index][0] >= rows[index + 1][0] for index in range(len(rows) - 1)):
            raise ValueError(f"duplicate or non-monotonic rows for {pair}")
        result[pair] = rows
    return result


def validate_source_corpus(root: Path, manifest_path: Path) -> dict[str, Any]:
    """Verify every M5 shard consumed by this experiment against the manifest."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("sources"), list):
        raise ValueError("source manifest has no sources inventory")
    resolved_root = root.resolve()
    if Path(manifest.get("source_root", "")).resolve() != resolved_root:
        raise ValueError("source root differs from manifest source_root")
    expected: dict[str, dict[str, Any]] = {}
    for row in manifest["sources"]:
        if not isinstance(row, dict) or row.get("pair") not in PAIRS:
            continue
        relative = row.get("relative_path")
        if not isinstance(relative, str) or relative in expected:
            raise ValueError("source manifest contains duplicate/invalid pair shard")
        expected[relative] = row
    actual = {
        str(path.relative_to(resolved_root))
        for pair in PAIRS
        for path in resolved_root.glob(f"*/{pair}/{pair}_M5_BA_*.jsonl.gz")
    }
    if actual != set(expected):
        missing = sorted(set(expected) - actual)
        extra = sorted(actual - set(expected))
        raise ValueError(
            f"source shard inventory mismatch missing={missing[:3]} extra={extra[:3]}"
        )
    inventory: list[dict[str, Any]] = []
    for relative in sorted(expected):
        row = expected[relative]
        path = resolved_root / relative
        size = path.stat().st_size
        if size != row.get("file_size_bytes"):
            raise ValueError(f"source shard size mismatch: {relative}")
        digest = raw_sha(path)
        if digest != row.get("file_sha256"):
            raise ValueError(f"source shard SHA-256 mismatch: {relative}")
        inventory.append(
            {
                "relative_path": relative,
                "file_size_bytes": size,
                "file_sha256": digest,
                "pair": row["pair"],
                "from_utc": row.get("from_utc"),
                "to_utc": row.get("to_utc"),
            }
        )
    body = {
        "source_root": str(resolved_root),
        "manifest_raw_sha256": raw_sha(manifest_path),
        "shard_count": len(inventory),
        "shards": inventory,
    }
    return {**body, "canonical_sha256": canonical_sha(body)}


def mid_close(row: tuple[Any, ...]) -> float:
    return (row[4] + row[8]) / 2.0


def _is_contiguous(
    rows: list[tuple[Any, ...]], start_index: int, end_epoch_inclusive: int
) -> bool:
    if start_index < 0 or start_index >= len(rows):
        return False
    first_epoch = rows[start_index][0]
    if end_epoch_inclusive < first_epoch:
        return False
    count = (end_epoch_inclusive - first_epoch) // 300 + 1
    if first_epoch + (count - 1) * 300 != end_epoch_inclusive:
        return False
    if start_index + count > len(rows):
        return False
    return all(
        rows[start_index + offset][0] == first_epoch + offset * 300
        for offset in range(count)
    )


def _tp_filled_before_decision(
    rows: list[tuple[Any, ...]],
    entry_bar: int,
    decision_bar: int,
    *,
    side: str,
    tp: float,
) -> bool:
    """Treat the entry-bar open as occurring before its later TP touch."""
    return any(
        (side == "LONG" and rows[index][2] >= tp)
        or (side == "SHORT" and rows[index][7] <= tp)
        for index in range(entry_bar, decision_bar)
    )


def triggered_signal(
    pair: str,
    rows: list[tuple[Any, ...]],
    index: dict[int, int],
    trigger_bar: int,
) -> tuple[str, float] | None:
    if trigger_bar < BREAK_BARS:
        return None
    row = rows[trigger_bar]
    close_epoch = row[0] + 300
    past_i = index.get(close_epoch - TREND_LOOKBACK_S - 300)
    if past_i is None:
        return None
    close = mid_close(row)
    past = mid_close(rows[past_i])
    side = "LONG" if close > past else "SHORT"
    window = rows[trigger_bar - BREAK_BARS : trigger_bar]
    if any(
        window[index + 1][0] - window[index][0] != 300
        for index in range(len(window) - 1)
    ):
        return None
    high = max((item[2] + item[6]) / 2.0 for item in window)
    low = min((item[3] + item[7]) / 2.0 for item in window)
    if side == "LONG" and close > high:
        strength = (close - high) / pip_size(pair)
    elif side == "SHORT" and close < low:
        strength = (low - close) / pip_size(pair)
    else:
        return None
    return side, round(strength, 9)


def completed_context(
    data: dict[str, list[tuple[Any, ...]]],
    indices: dict[str, dict[int, int]],
    pair: str,
    decision_bar: int,
) -> tuple[list[float], list[list[float]], float]:
    rows = data[pair]
    decision_epoch = rows[decision_bar][0]
    if decision_bar < M5_BARS:
        raise ValueError("insufficient M5 context")
    m5_window = rows[decision_bar - M5_BARS : decision_bar]
    if (
        len(m5_window) != M5_BARS
        or any(row[0] >= decision_epoch for row in m5_window)
        or any(
            m5_window[index + 1][0] - m5_window[index][0] != 300
            for index in range(len(m5_window) - 1)
        )
        or m5_window[-1][0] != decision_epoch - 300
    ):
        raise ValueError("M5 context is incomplete or crosses decision open")

    h1_rev: list[float] = []
    hour_anchor = (decision_epoch // 3600) * 3600
    pair_index = indices[pair]
    for hour_offset in range(1, 400):
        if len(h1_rev) >= H1_BARS:
            break
        target = hour_anchor - hour_offset * 3600 + 3600 - 300
        source_i = pair_index.get(target)
        if source_i is not None:
            h1_rev.append(mid_close(rows[source_i]))
    if len(h1_rev) != H1_BARS:
        raise ValueError("insufficient completed H1 context")
    h1 = list(reversed(h1_rev))
    base = h1[0]
    if base <= 0:
        raise ValueError("invalid normalization base")

    def norm(value: float) -> float:
        return round(value / base * 100.0, 4)

    return (
        [norm(value) for value in h1],
        [
            [
                norm((row[1] + row[5]) / 2.0),
                norm((row[2] + row[6]) / 2.0),
                norm((row[3] + row[7]) / 2.0),
                norm((row[4] + row[8]) / 2.0),
            ]
            for row in m5_window
        ],
        base,
    )


def build_cases(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build deterministic packets without calculating either branch outcome."""
    data = load_data(root)
    indices = {
        pair: {row[0]: index for index, row in enumerate(rows)}
        for pair, rows in data.items()
    }
    epochs = {pair: [row[0] for row in rows] for pair, rows in data.items()}
    rng = random.Random(SEED)
    cases: list[dict[str, Any]] = []
    attempts = 0
    while len(cases) < CELL_COUNT and attempts < 100_000:
        attempts += 1
        pair = rng.choice(PAIRS)
        rows = data[pair]
        lower = max(
            bisect_left(epochs[pair], int(FROM.timestamp())),
            H1_BARS * 12 + M5_BARS + 10,
        )
        upper = min(
            bisect_left(epochs[pair], int(TO.timestamp())),
            len(rows) - (DECISION_DELAY_S + HOLD_HORIZON_S) // 300 - 2,
        )
        if lower >= upper:
            raise ValueError(f"insufficient 2026 H1 data for {pair}")
        trigger_bar = rng.randrange(lower, upper)
        trigger_stamp = datetime.fromtimestamp(rows[trigger_bar][0], tz=UTC)
        status = compute_market_status(trigger_stamp)
        if (
            not status.is_fx_open
            or status.minutes_to_next_close is None
            or status.minutes_to_next_close
            <= (DECISION_DELAY_S + HOLD_HORIZON_S) / 60 + 30
        ):
            continue
        signal = triggered_signal(pair, rows, indices[pair], trigger_bar)
        if signal is None:
            continue
        side, _ = signal
        if (
            trigger_bar + 1 >= len(rows)
            or rows[trigger_bar + 1][0] != rows[trigger_bar][0] + 300
        ):
            continue
        entry_bar = trigger_bar + 1
        entry_epoch = rows[entry_bar][0]
        pip = pip_size(pair)
        entry = rows[entry_bar][5] if side == "LONG" else rows[entry_bar][1]
        tp = entry + TP_PIPS * pip if side == "LONG" else entry - TP_PIPS * pip
        target_epoch = entry_epoch + DECISION_DELAY_S
        decision_bar = bisect_left(epochs[pair], target_epoch)
        if decision_bar >= len(rows) or rows[decision_bar][0] != target_epoch:
            continue
        decision_epoch = rows[decision_bar][0]
        horizon_end_epoch = decision_epoch + HOLD_HORIZON_S
        if not (FROM.timestamp() <= decision_epoch < TO.timestamp()):
            continue
        if horizon_end_epoch >= TO.timestamp():
            continue
        if any(
            abs(decision_epoch - case["private"]["decision_epoch"])
            < MIN_CASE_SEPARATION_S
            for case in cases
        ):
            continue
        # Future timestamps are checked only for source continuity. No future
        # price or branch result is read for selection.
        if not _is_contiguous(rows, entry_bar, horizon_end_epoch):
            continue
        filled_before_decision = _tp_filled_before_decision(
            rows,
            entry_bar,
            decision_bar,
            side=side,
            tp=tp,
        )
        if filled_before_decision:
            continue

        candidates: list[tuple[float, str, int, str]] = []
        for candidate_pair in PAIRS:
            if candidate_pair == pair:
                continue
            candidate_i = indices[candidate_pair].get(decision_epoch)
            if candidate_i is None or candidate_i < 1:
                continue
            candidate_signal = triggered_signal(
                candidate_pair,
                data[candidate_pair],
                indices[candidate_pair],
                candidate_i - 1,
            )
            if candidate_signal is None:
                continue
            candidate_side, strength = candidate_signal
            if not _is_contiguous(data[candidate_pair], candidate_i, horizon_end_epoch):
                continue
            candidates.append((strength, candidate_pair, candidate_i, candidate_side))
        if not candidates:
            continue
        # This ranking is based only on the causal break observed before the
        # decision open. It does not inspect either candidate outcome.
        _, candidate_pair, candidate_bar, candidate_side = max(
            candidates, key=lambda item: (item[0], item[1])
        )
        try:
            a_h1, a_m5, a_base = completed_context(data, indices, pair, decision_bar)
            b_h1, b_m5, b_base = completed_context(
                data, indices, candidate_pair, candidate_bar
            )
        except ValueError:
            continue

        cut_price = rows[decision_bar][1] if side == "LONG" else rows[decision_bar][5]
        candidate_rows = data[candidate_pair]
        candidate_entry = (
            candidate_rows[candidate_bar][5]
            if candidate_side == "LONG"
            else candidate_rows[candidate_bar][1]
        )
        candidate_pip = pip_size(candidate_pair)
        candidate_tp = (
            candidate_entry + TP_PIPS * candidate_pip
            if candidate_side == "LONG"
            else candidate_entry - TP_PIPS * candidate_pip
        )
        case_id = f"P{len(cases) + 1:02d}"
        packet_body = {
            "id": case_id,
            "capacity_contract": "FIXED_HIERARCHICAL_GATE; NO_MODEL_ALLOCATION",
            "decision_price_basis": "DECISION_BAR_OPEN",
            "context_cutoff": "STRICTLY_BEFORE_DECISION_BAR",
            "horizon": "TP_OR_HARD_EXIT_PLUS_4H",
            "asset_a_existing": {
                "side": side,
                "minutes_held": int((decision_epoch - entry_epoch) / 60),
                "entry_norm": round(entry / a_base * 100.0, 4),
                "tp_norm": round(tp / a_base * 100.0, 4),
                "current_norm": round(cut_price / a_base * 100.0, 4),
                "unrealized_pips": round(
                    (cut_price - entry) / pip
                    if side == "LONG"
                    else (entry - cut_price) / pip,
                    1,
                ),
                "h1_closes_norm": a_h1,
                "m5_ohlc_norm": a_m5,
            },
            "asset_b_new_candidate": {
                "side": candidate_side,
                "trigger": "24H_TREND_3BAR_M5_BREAK_NEXT_OPEN",
                "entry_norm": round(candidate_entry / b_base * 100.0, 4),
                "tp_norm": round(candidate_tp / b_base * 100.0, 4),
                "h1_closes_norm": b_h1,
                "m5_ohlc_norm": b_m5,
            },
        }
        packet = {
            **packet_body,
            "packet_canonical_sha256": canonical_sha(packet_body),
        }
        cases.append(
            {
                "packet": packet,
                "private": {
                    "existing_pair": pair,
                    "existing_side": side,
                    "entry": entry,
                    "entry_epoch": entry_epoch,
                    "tp": tp,
                    "pip": pip,
                    "decision_bar": decision_bar,
                    "decision_epoch": decision_epoch,
                    "candidate_pair": candidate_pair,
                    "candidate_side": candidate_side,
                    "candidate_entry": candidate_entry,
                    "candidate_tp": candidate_tp,
                    "candidate_pip": candidate_pip,
                    "candidate_bar": candidate_bar,
                },
            }
        )
    if len(cases) != CELL_COUNT:
        raise ValueError(f"only found {len(cases)} cases after {attempts} attempts")
    return cases, {"data": data, "attempts": attempts}


def outcome(
    rows: list[tuple[Any, ...]],
    *,
    decision_bar: int,
    side: str,
    entry: float,
    tp: float,
    pip: float,
) -> tuple[float, float]:
    """Return immediate-cut and post-decision-hold pips after response sealing."""
    decision_epoch = rows[decision_bar][0]
    cut_price = rows[decision_bar][1] if side == "LONG" else rows[decision_bar][5]
    cut_pips = (
        (cut_price - entry) / pip if side == "LONG" else (entry - cut_price) / pip
    )
    end_epoch = decision_epoch + HOLD_HORIZON_S
    index = decision_bar
    held_pips: float | None = None
    while index < len(rows) and rows[index][0] < end_epoch:
        row = rows[index]
        if (side == "LONG" and row[2] >= tp) or (side == "SHORT" and row[7] <= tp):
            held_pips = TP_PIPS
            break
        index += 1
    if held_pips is None:
        if index >= len(rows) or rows[index][0] != end_epoch:
            raise ValueError("outcome horizon unavailable or discontinuous")
        end_price = rows[index][1] if side == "LONG" else rows[index][5]
        held_pips = (
            (end_price - entry) / pip if side == "LONG" else (entry - end_price) / pip
        )
    return round(cut_pips, 2), round(held_pips, 2)


def _validate_packet(packet: dict[str, Any]) -> None:
    body = dict(packet)
    claimed = body.pop("packet_canonical_sha256", None)
    if not isinstance(claimed, str) or canonical_sha(body) != claimed:
        raise ValueError(f"packet canonical SHA-256 mismatch for {packet.get('id')}")


def _load_and_verify_prepared(out: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    prereg = json.loads((out / "preregistration.json").read_text(encoding="utf-8"))
    prereg_body = _body_with_verified_canonical_sha(prereg, "preregistration")
    packets = json.loads((out / "packets.json").read_text(encoding="utf-8"))
    packets_body = _body_with_verified_canonical_sha(packets, "packets")
    if prereg_body["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("unexpected experiment id")
    if packets_body["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("packet experiment id mismatch")
    if packets_body.get("answer_key_present") is not False:
        raise ValueError("prepared packet set claims an answer key")
    packet_rows = packets_body.get("packets")
    if not isinstance(packet_rows, list) or len(packet_rows) != CELL_COUNT:
        raise ValueError("packet set must contain exactly six packets")
    for packet in packet_rows:
        _validate_packet(packet)
    bindings = prereg_body["bindings"]
    if raw_sha(Path(__file__)) != bindings["experiment_code_raw_sha256"]:
        raise ValueError("experiment code changed after preregistration")
    if raw_sha(out / "prompt_template.txt") != bindings["prompt_template_raw_sha256"]:
        raise ValueError("prompt changed after preregistration")
    if raw_sha(out / "packets.json") != bindings["packets_raw_sha256"]:
        raise ValueError("packets changed after preregistration")
    if packets["canonical_sha256"] != bindings["packets_canonical_sha256"]:
        raise ValueError("packet canonical binding mismatch")
    return prereg, packets


def prepare(args: argparse.Namespace) -> int:
    out = args.out_dir
    guarded_names = (
        "preregistration.json",
        "packets.json",
        "prompt_template.txt",
        "response_manifest.json",
        "answer_key.json",
        "evidence.json",
    )
    for name in guarded_names:
        if (out / name).exists():
            raise ValueError(f"refusing to overwrite {name}")
    if (out / "responses").exists() or (out / "raw-responses").exists():
        raise ValueError("refusing to prepare over response directories")
    corpus = validate_source_corpus(args.root, args.source_manifest)
    cases, meta = build_cases(args.root)
    packets_body = {
        "contract": "QR_DOJO_CAPITAL_RECYCLE_PACKETS_V2",
        "experiment_id": EXPERIMENT_ID,
        "seed": SEED,
        "cell_count": CELL_COUNT,
        "answer_key_present": False,
        "packets": [case["packet"] for case in cases],
    }
    packets = {**packets_body, "canonical_sha256": canonical_sha(packets_body)}
    atomic_json(out / "packets.json", packets)
    atomic_text(out / "prompt_template.txt", PROMPT_TEMPLATE)
    prereg_body = {
        "contract": "QR_DOJO_CAPITAL_RECYCLE_PREREGISTRATION_V2",
        "experiment_id": EXPERIMENT_ID,
        "recorded_at_utc": datetime.now(tz=UTC).isoformat(),
        "classification": "WORN_HISTORY_TRAIN_DIAGNOSTIC",
        "proof_eligible": False,
        "promotion_eligible": False,
        "live_permission": False,
        "order_authority": "NONE",
        "question": "Does a fixed KEEP/HOLD, FLAT/recycle hierarchy improve capital use?",
        "differential_task": "EXIT_GATE_THEN_BINARY_CAPITAL_RECYCLE",
        "cell_count": CELL_COUNT,
        "seed": SEED,
        "selection": (
            "SIX_SEEDED_24H_SEPARATED_EXISTING_POSITIONS_WITH_CAUSAL_"
            "SIMULTANEOUS_BURST_CANDIDATE; MAX_CAUSAL_BREAK_STRENGTH"
        ),
        "one_judgment_per_fresh_context": True,
        "answer_key_generation": "SCORE_MODE_ONLY_AFTER_EXACT_SIX_CELL_SEALS_AND_FINAL_MANIFEST",
        "answer_key_present_at_prepare": False,
        "model_policy": {
            "model_id": args.model_id,
            "model_lineage": args.model_lineage,
            "provider_attestation_required": "SELF_ATTESTED_NO_PROVIDER_SIGNATURE",
            "fork_turns_none_required": True,
            "tools_used_required": False,
        },
        "invalid_or_missing_response_policy": {
            "fixed_denominator": CELL_COUNT,
            "action": "SYNTHETIC_FAILURE_FLAT_EXISTING_FLAT_NEXT_CUT_TO_RESERVE",
            "retry_allowed": False,
            "seal_missing_before_truth_allowed": True,
            "deadline_externally_attested": False,
        },
        "model_allocation_discretion": False,
        "fixed_policy": {
            "gate_order": ["EXISTING_DIRECTION", "NEXT_DIRECTION"],
            "existing_keep": "100_PERCENT_EXISTING_FULL_HOLD",
            "existing_flat_next_valid": "0_PERCENT_EXISTING_100_PERCENT_NEXT",
            "existing_flat_next_flat": "0_PERCENT_EXISTING_0_PERCENT_NEXT_100_PERCENT_RESERVE",
            "partial_allocations_allowed": False,
        },
        "decision_horizon_hours": 4,
        "cost_model": {
            "bid_ask_spread_embedded": True,
            "additional_slippage_pips": 0.0,
            "financing_pips": 0.0,
            "all_costs_proved": False,
        },
        "scoring": {
            "hierarchical_policy": "KEEP=>HOLD; FLAT+VALID_NEXT=>CUT+NEXT; FLAT+FLAT=>CUT",
            "comparators": [
                "FULL_HOLD",
                "CUT_TO_RESERVE",
                "ROTATE_FULL_TO_NEXT",
                "FULL_ALLOCATION_ORACLE",
            ],
            "direction_truth": "existing KEEP iff hold>cut; next valid iff next>0",
            "oracle_actions": ["FULL_HOLD", "CUT_TO_RESERVE", "ROTATE_FULL_TO_NEXT"],
        },
        "source": {
            "root": str(args.root.resolve()),
            "manifest_path": str(args.source_manifest.resolve()),
            "manifest_raw_sha256": raw_sha(args.source_manifest),
            "corpus_inventory_canonical_sha256": corpus["canonical_sha256"],
            "corpus_shard_count": corpus["shard_count"],
            "granularity": "M5",
            "price_component": "BA",
            "period_from_inclusive_utc": FROM.isoformat(),
            "period_to_exclusive_utc": TO.isoformat(),
            "all_case_horizons_end_before_period_to": True,
            "prior_v1_period_to_exclusive_utc": "2025-12-01T00:00:00+00:00",
        },
        "bindings": {
            "experiment_code_raw_sha256": raw_sha(Path(__file__)),
            "prompt_template_raw_sha256": raw_sha(out / "prompt_template.txt"),
            "packets_raw_sha256": raw_sha(out / "packets.json"),
            "packets_canonical_sha256": packets["canonical_sha256"],
        },
        "sample_attempt_count": meta["attempts"],
    }
    prereg = {**prereg_body, "canonical_sha256": canonical_sha(prereg_body)}
    atomic_json(out / "preregistration.json", prereg)
    print(json.dumps({"status": "PREPARED_WITHOUT_ANSWER_KEY", "cells": CELL_COUNT}))
    return 0


def validate_decision(packet: dict[str, Any], row: dict[str, Any]) -> None:
    expected_keys = {
        "id",
        "existing_direction",
        "next_direction",
        "conviction",
        "reason",
    }
    if set(row) != expected_keys or row["id"] != packet["id"]:
        raise ValueError(f"invalid response shape/id for {packet['id']}")
    a_side = packet["asset_a_existing"]["side"]
    b_side = packet["asset_b_new_candidate"]["side"]
    if row["existing_direction"] not in {f"KEEP_{a_side}", "FLAT"}:
        raise ValueError(f"invalid existing direction for {packet['id']}")
    if row["next_direction"] not in {b_side, "FLAT"}:
        raise ValueError(f"invalid next direction for {packet['id']}")
    if (
        isinstance(row["conviction"], bool)
        or not isinstance(row["conviction"], int)
        or row["conviction"] not in {1, 2, 3}
    ):
        raise ValueError(f"invalid conviction for {packet['id']}")
    if not isinstance(row["reason"], str) or len(row["reason"]) > 240:
        raise ValueError(f"invalid reason for {packet['id']}")


def _packet_map(packets: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {packet["id"]: packet for packet in packets["packets"]}


def _expected_cell_bindings(
    out: Path,
    prereg: dict[str, Any],
    packet: dict[str, Any],
) -> dict[str, Any]:
    prereg_body = _body_with_verified_canonical_sha(prereg, "preregistration")
    return {
        "preregistration_raw_sha256": raw_sha(out / "preregistration.json"),
        "preregistration_canonical_sha256": prereg["canonical_sha256"],
        "packets_raw_sha256": raw_sha(out / "packets.json"),
        "prompt_template_raw_sha256": raw_sha(out / "prompt_template.txt"),
        "experiment_code_raw_sha256": raw_sha(Path(__file__)),
        "packet_canonical_sha256": packet["packet_canonical_sha256"],
        "source_corpus_inventory_canonical_sha256": prereg_body["source"][
            "corpus_inventory_canonical_sha256"
        ],
    }


def _validate_execution(
    execution: dict[str, Any], model_policy: dict[str, Any]
) -> None:
    expected_keys = {
        "context_id",
        "agent_receipt_id",
        "model_id",
        "model_lineage",
        "provider_attestation",
        "fork_turns_none",
        "tools_used",
    }
    if not isinstance(execution, dict) or set(execution) != expected_keys:
        raise ValueError("invalid per-cell execution provenance shape")
    for key in ("context_id", "agent_receipt_id"):
        if not isinstance(execution[key], str) or not execution[key].strip():
            raise ValueError(f"invalid per-cell {key}")
    if execution["model_id"] != model_policy["model_id"]:
        raise ValueError("per-cell model id differs from preregistration")
    if execution["model_lineage"] != model_policy["model_lineage"]:
        raise ValueError("per-cell model lineage differs from preregistration")
    if (
        execution["provider_attestation"]
        != model_policy["provider_attestation_required"]
    ):
        raise ValueError("per-cell provider attestation differs from preregistration")
    if execution["fork_turns_none"] is not True:
        raise ValueError("per-cell context was not fork_turns=none")
    if execution["tools_used"] is not False:
        raise ValueError("per-cell context used forbidden tools")


def _terminal_body(
    *,
    out: Path,
    prereg: dict[str, Any],
    packet: dict[str, Any],
    status: str,
    decision: dict[str, Any] | None,
    execution: dict[str, Any] | None,
    validation_error: str | None,
    raw_response_path: Path | None,
) -> dict[str, Any]:
    return {
        "contract": "QR_DOJO_CAPITAL_RECYCLE_CELL_TERMINAL_V2",
        "experiment_id": EXPERIMENT_ID,
        "id": packet["id"],
        "sealed_at_utc": datetime.now(tz=UTC).isoformat(),
        "status": status,
        "decision": decision,
        "execution": execution,
        "validation_error": validation_error,
        "effective_action": (
            _policy_action(decision)
            if status == "VALID" and decision
            else "CUT_TO_RESERVE"
        ),
        "direction_hits_count_as_false": status != "VALID",
        "raw_response_relative_path": (
            str(raw_response_path.relative_to(out)) if raw_response_path else None
        ),
        "raw_response_raw_sha256": (
            raw_sha(raw_response_path) if raw_response_path else None
        ),
        "bindings": _expected_cell_bindings(out, prereg, packet),
    }


def seal_cell(args: argparse.Namespace) -> int:
    """Preserve one raw first attempt, then seal VALID or synthetic failure."""
    out = args.out_dir
    prereg, packets = _load_and_verify_prepared(out)
    packet = _packet_map(packets).get(args.cell_id)
    if packet is None:
        raise ValueError(f"unknown cell id: {args.cell_id}")
    terminal_path = out / "responses" / f"{args.cell_id}.json"
    raw_path = out / "raw-responses" / f"{args.cell_id}.json"
    if terminal_path.exists() or raw_path.exists():
        raise ValueError(f"refusing to overwrite first attempt for {args.cell_id}")
    raw_bytes = args.response.read_bytes()
    # O_EXCL publishes the exact first-attempt bytes before schema inspection.
    first_write_bytes(raw_path, raw_bytes)
    decision: dict[str, Any] | None = None
    execution: dict[str, Any] | None = None
    status = "VALID"
    validation_error: str | None = None
    try:
        source = json.loads(raw_bytes)
        if not isinstance(source, dict) or set(source) != {
            "contract",
            "experiment_id",
            "id",
            "execution",
            "decision",
        }:
            raise ValueError("raw cell response envelope has invalid fields")
        if source["contract"] != "QR_DOJO_CAPITAL_RECYCLE_RAW_CELL_RESPONSE_V2":
            raise ValueError("unexpected raw cell response contract")
        if source["experiment_id"] != EXPERIMENT_ID or source["id"] != args.cell_id:
            raise ValueError("raw cell response identity mismatch")
        execution = source["execution"]
        prereg_body = _body_with_verified_canonical_sha(prereg, "preregistration")
        _validate_execution(execution, prereg_body["model_policy"])
        decision = source["decision"]
        validate_decision(packet, decision)
    except Exception as exc:
        status = "SYNTHETIC_FAILURE"
        decision = None
        validation_error = f"{type(exc).__name__}: {exc}"
    body = _terminal_body(
        out=out,
        prereg=prereg,
        packet=packet,
        status=status,
        decision=decision,
        execution=execution if isinstance(execution, dict) else None,
        validation_error=validation_error,
        raw_response_path=raw_path,
    )
    terminal = {**body, "canonical_sha256": canonical_sha(body)}
    atomic_json(terminal_path, terminal)
    print(json.dumps({"status": status, "id": args.cell_id}))
    return 0


def seal_missing_cell(args: argparse.Namespace) -> int:
    """Irreversibly terminalize one absent response as fixed-denominator failure."""
    out = args.out_dir
    prereg, packets = _load_and_verify_prepared(out)
    packet = _packet_map(packets).get(args.cell_id)
    if packet is None:
        raise ValueError(f"unknown cell id: {args.cell_id}")
    terminal_path = out / "responses" / f"{args.cell_id}.json"
    raw_path = out / "raw-responses" / f"{args.cell_id}.json"
    if terminal_path.exists() or raw_path.exists():
        raise ValueError(f"refusing to overwrite first attempt for {args.cell_id}")
    body = _terminal_body(
        out=out,
        prereg=prereg,
        packet=packet,
        status="SYNTHETIC_MISSING",
        decision=None,
        execution=None,
        validation_error="NO_RESPONSE_SEALED_BEFORE_TRUTH; NO_EXTERNAL_DEADLINE_ATTESTATION",
        raw_response_path=None,
    )
    terminal = {**body, "canonical_sha256": canonical_sha(body)}
    atomic_json(terminal_path, terminal)
    print(json.dumps({"status": "SYNTHETIC_MISSING", "id": args.cell_id}))
    return 0


def _verify_cell_terminal(
    out: Path,
    prereg: dict[str, Any],
    packet: dict[str, Any],
    path: Path,
) -> dict[str, Any]:
    terminal = json.loads(path.read_text(encoding="utf-8"))
    body = _body_with_verified_canonical_sha(terminal, f"cell terminal {packet['id']}")
    if body.get("id") != packet["id"] or path.name != f"{packet['id']}.json":
        raise ValueError("cell terminal filename/id mismatch")
    if body.get("bindings") != _expected_cell_bindings(out, prereg, packet):
        raise ValueError(f"cell terminal bindings mismatch for {packet['id']}")
    status = body.get("status")
    if status not in {"VALID", "SYNTHETIC_FAILURE", "SYNTHETIC_MISSING"}:
        raise ValueError(f"invalid cell terminal status for {packet['id']}")
    raw_relative = body.get("raw_response_relative_path")
    raw_digest = body.get("raw_response_raw_sha256")
    expected_raw = out / "raw-responses" / f"{packet['id']}.json"
    if status == "SYNTHETIC_MISSING":
        if raw_relative is not None or raw_digest is not None or expected_raw.exists():
            raise ValueError(f"missing terminal has raw response for {packet['id']}")
    else:
        if (
            raw_relative != str(expected_raw.relative_to(out))
            or not expected_raw.exists()
        ):
            raise ValueError(f"raw response path missing for {packet['id']}")
        if raw_digest != raw_sha(expected_raw):
            raise ValueError(f"raw response SHA-256 mismatch for {packet['id']}")
    if status == "VALID":
        prereg_body = _body_with_verified_canonical_sha(prereg, "preregistration")
        _validate_execution(body.get("execution"), prereg_body["model_policy"])
        validate_decision(packet, body.get("decision"))
        if body.get("effective_action") != _policy_action(body["decision"]):
            raise ValueError(f"effective action mismatch for {packet['id']}")
    elif body.get("effective_action") != "CUT_TO_RESERVE":
        raise ValueError(f"synthetic failure action mismatch for {packet['id']}")
    return terminal


def finalize_responses(args: argparse.Namespace) -> int:
    out = args.out_dir
    target = out / "response_manifest.json"
    if target.exists():
        raise ValueError("refusing to overwrite response_manifest.json")
    prereg, packets = _load_and_verify_prepared(out)
    packet_by_id = _packet_map(packets)
    expected_paths = {out / "responses" / f"{cell_id}.json" for cell_id in packet_by_id}
    actual_paths = set((out / "responses").glob("*.json"))
    if actual_paths != expected_paths:
        missing = sorted(path.name for path in expected_paths - actual_paths)
        extra = sorted(path.name for path in actual_paths - expected_paths)
        raise ValueError(
            f"exact six cell terminals required missing={missing} extra={extra}"
        )
    terminals = {
        cell_id: _verify_cell_terminal(
            out, prereg, packet, out / "responses" / f"{cell_id}.json"
        )
        for cell_id, packet in packet_by_id.items()
    }
    valid_contexts = [
        terminal["execution"]["context_id"]
        for terminal in terminals.values()
        if terminal["status"] == "VALID"
    ]
    valid_receipts = [
        terminal["execution"]["agent_receipt_id"]
        for terminal in terminals.values()
        if terminal["status"] == "VALID"
    ]
    duplicate_contexts = {
        value for value in valid_contexts if valid_contexts.count(value) > 1
    }
    duplicate_receipts = {
        value for value in valid_receipts if valid_receipts.count(value) > 1
    }
    rows: list[dict[str, Any]] = []
    for cell_id in sorted(packet_by_id):
        terminal = terminals[cell_id]
        effective_status = terminal["status"]
        effective_action = terminal["effective_action"]
        execution = terminal.get("execution")
        if terminal["status"] == "VALID" and (
            execution["context_id"] in duplicate_contexts
            or execution["agent_receipt_id"] in duplicate_receipts
        ):
            effective_status = "SYNTHETIC_FAILURE_DUPLICATE_CONTEXT"
            effective_action = "CUT_TO_RESERVE"
        terminal_path = out / "responses" / f"{cell_id}.json"
        rows.append(
            {
                "id": cell_id,
                "packet_canonical_sha256": packet_by_id[cell_id][
                    "packet_canonical_sha256"
                ],
                "terminal_relative_path": str(terminal_path.relative_to(out)),
                "terminal_raw_sha256": raw_sha(terminal_path),
                "terminal_canonical_sha256": terminal["canonical_sha256"],
                "raw_response_raw_sha256": terminal["raw_response_raw_sha256"],
                "sealed_status": terminal["status"],
                "effective_status": effective_status,
                "effective_action": effective_action,
            }
        )
    manifest_body = {
        "contract": "QR_DOJO_CAPITAL_RECYCLE_RESPONSE_MANIFEST_V2",
        "experiment_id": EXPERIMENT_ID,
        "finalized_at_utc": datetime.now(tz=UTC).isoformat(),
        "fixed_denominator": CELL_COUNT,
        "cell_count": len(rows),
        "bindings": {
            "preregistration_raw_sha256": raw_sha(out / "preregistration.json"),
            "preregistration_canonical_sha256": prereg["canonical_sha256"],
            "packets_raw_sha256": raw_sha(out / "packets.json"),
            "prompt_template_raw_sha256": raw_sha(out / "prompt_template.txt"),
            "experiment_code_raw_sha256": raw_sha(Path(__file__)),
        },
        "cells": rows,
    }
    manifest = {**manifest_body, "canonical_sha256": canonical_sha(manifest_body)}
    atomic_json(target, manifest)
    print(json.dumps({"status": "RESPONSES_FINALIZED", "cells": len(rows)}))
    return 0


def _policy_action(decision: dict[str, Any]) -> str:
    if decision["existing_direction"].startswith("KEEP_"):
        return "FULL_HOLD"
    if decision["next_direction"] != "FLAT":
        return "ROTATE_FULL_TO_NEXT"
    return "CUT_TO_RESERVE"


def score(args: argparse.Namespace) -> int:
    out = args.out_dir
    result_path = out / "evidence.json"
    answer_path = out / "answer_key.json"
    if result_path.exists() or answer_path.exists():
        raise ValueError("refusing to overwrite score artifacts")
    prereg, packets = _load_and_verify_prepared(out)
    prereg_body = _body_with_verified_canonical_sha(prereg, "preregistration")
    prereg_root = Path(prereg_body["source"]["root"])
    prereg_manifest = Path(prereg_body["source"]["manifest_path"])
    if args.root.resolve() != prereg_root.resolve():
        raise ValueError("score root differs from preregistered source root")
    if args.source_manifest.resolve() != prereg_manifest.resolve():
        raise ValueError("score manifest differs from preregistered source manifest")
    if raw_sha(prereg_manifest) != prereg_body["source"]["manifest_raw_sha256"]:
        raise ValueError("source manifest changed after preregistration")
    corpus = validate_source_corpus(args.root, args.source_manifest)
    if (
        corpus["canonical_sha256"]
        != prereg_body["source"]["corpus_inventory_canonical_sha256"]
    ):
        raise ValueError("source corpus differs from preregistered shard inventory")
    response_manifest_path = out / "response_manifest.json"
    if not response_manifest_path.exists():
        raise ValueError(
            "response_manifest.json must finalize exact six seals before scoring"
        )
    response_manifest = json.loads(response_manifest_path.read_text(encoding="utf-8"))
    response_manifest_body = _body_with_verified_canonical_sha(
        response_manifest, "response manifest"
    )
    bindings = response_manifest_body["bindings"]
    expected_bindings = {
        "preregistration_raw_sha256": raw_sha(out / "preregistration.json"),
        "preregistration_canonical_sha256": prereg["canonical_sha256"],
        "packets_raw_sha256": raw_sha(out / "packets.json"),
        "prompt_template_raw_sha256": raw_sha(out / "prompt_template.txt"),
        "experiment_code_raw_sha256": raw_sha(Path(__file__)),
    }
    if bindings != expected_bindings:
        raise ValueError("response manifest bindings mismatch")
    manifest_rows = response_manifest_body.get("cells")
    if (
        response_manifest_body.get("fixed_denominator") != CELL_COUNT
        or response_manifest_body.get("cell_count") != CELL_COUNT
        or not isinstance(manifest_rows, list)
        or len(manifest_rows) != CELL_COUNT
    ):
        raise ValueError("response manifest violates fixed six-cell denominator")
    packet_by_id = _packet_map(packets)
    manifest_by_id = {
        row.get("id"): row for row in manifest_rows if isinstance(row, dict)
    }
    if set(manifest_by_id) != set(packet_by_id):
        raise ValueError("response manifest cell ids do not match packet ids")
    terminals: dict[str, dict[str, Any]] = {}
    for cell_id, packet in packet_by_id.items():
        row = manifest_by_id[cell_id]
        expected_relative = f"responses/{cell_id}.json"
        if row.get("terminal_relative_path") != expected_relative:
            raise ValueError(f"terminal path mismatch for {cell_id}")
        terminal_path = out / expected_relative
        if raw_sha(terminal_path) != row.get("terminal_raw_sha256"):
            raise ValueError(f"terminal raw SHA-256 mismatch for {cell_id}")
        terminal = _verify_cell_terminal(out, prereg, packet, terminal_path)
        if terminal["canonical_sha256"] != row.get("terminal_canonical_sha256"):
            raise ValueError(f"terminal canonical SHA-256 mismatch for {cell_id}")
        if terminal["raw_response_raw_sha256"] != row.get("raw_response_raw_sha256"):
            raise ValueError(f"manifest raw response SHA-256 mismatch for {cell_id}")
        if row.get("sealed_status") != terminal["status"]:
            raise ValueError(f"terminal status mismatch for {cell_id}")
        if row.get("effective_status") not in {
            "VALID",
            "SYNTHETIC_FAILURE",
            "SYNTHETIC_MISSING",
            "SYNTHETIC_FAILURE_DUPLICATE_CONTEXT",
        }:
            raise ValueError(f"invalid effective status for {cell_id}")
        if row.get("effective_action") not in {
            "FULL_HOLD",
            "CUT_TO_RESERVE",
            "ROTATE_FULL_TO_NEXT",
        }:
            raise ValueError(f"invalid effective action for {cell_id}")
        terminals[cell_id] = terminal

    valid_contexts = [
        terminal["execution"]["context_id"]
        for terminal in terminals.values()
        if terminal["status"] == "VALID"
    ]
    valid_receipts = [
        terminal["execution"]["agent_receipt_id"]
        for terminal in terminals.values()
        if terminal["status"] == "VALID"
    ]
    duplicate_contexts = {
        value for value in valid_contexts if valid_contexts.count(value) > 1
    }
    duplicate_receipts = {
        value for value in valid_receipts if valid_receipts.count(value) > 1
    }
    for cell_id, terminal in terminals.items():
        expected_status = terminal["status"]
        expected_action = terminal["effective_action"]
        execution = terminal.get("execution")
        if terminal["status"] == "VALID" and (
            execution["context_id"] in duplicate_contexts
            or execution["agent_receipt_id"] in duplicate_receipts
        ):
            expected_status = "SYNTHETIC_FAILURE_DUPLICATE_CONTEXT"
            expected_action = "CUT_TO_RESERVE"
        row = manifest_by_id[cell_id]
        if row["effective_status"] != expected_status:
            raise ValueError(
                f"effective status was not deterministically derived for {cell_id}"
            )
        if row["effective_action"] != expected_action:
            raise ValueError(
                f"effective action was not deterministically derived for {cell_id}"
            )

    cases, _ = build_cases(args.root)
    if [case["packet"] for case in cases] != packets["packets"]:
        raise ValueError("reconstructed packets differ from sealed packets")
    data = load_data(args.root)
    cells: list[dict[str, Any]] = []
    answer_rows: list[dict[str, Any]] = []
    totals = {
        "hierarchical": 0.0,
        "full_hold": 0.0,
        "cut_to_reserve": 0.0,
        "rotate_full_to_next": 0.0,
        "full_allocation_oracle": 0.0,
    }
    existing_hits = next_hits = joint_hits = 0
    for case in cases:
        packet = case["packet"]
        private = case["private"]
        terminal = terminals[packet["id"]]
        manifest_row = manifest_by_id[packet["id"]]
        decision = terminal["decision"]
        effective_valid = manifest_row["effective_status"] == "VALID"
        cut_pips, hold_pips = outcome(
            data[private["existing_pair"]],
            decision_bar=private["decision_bar"],
            side=private["existing_side"],
            entry=private["entry"],
            tp=private["tp"],
            pip=private["pip"],
        )
        _, next_pips = outcome(
            data[private["candidate_pair"]],
            decision_bar=private["candidate_bar"],
            side=private["candidate_side"],
            entry=private["candidate_entry"],
            tp=private["candidate_tp"],
            pip=private["candidate_pip"],
        )
        action_values = {
            "FULL_HOLD": hold_pips,
            "CUT_TO_RESERVE": cut_pips,
            "ROTATE_FULL_TO_NEXT": round(cut_pips + next_pips, 2),
        }
        selected_action = manifest_row["effective_action"]
        hierarchical_pips = action_values[selected_action]
        oracle_action, oracle_pips = max(
            action_values.items(), key=lambda item: item[1]
        )
        existing_truth = (
            f"KEEP_{private['existing_side']}" if hold_pips > cut_pips else "FLAT"
        )
        next_truth = private["candidate_side"] if next_pips > 0 else "FLAT"
        existing_hit = bool(
            effective_valid and decision["existing_direction"] == existing_truth
        )
        next_hit = bool(effective_valid and decision["next_direction"] == next_truth)
        existing_hits += int(existing_hit)
        next_hits += int(next_hit)
        joint_hits += int(existing_hit and next_hit)
        totals["hierarchical"] += hierarchical_pips
        totals["full_hold"] += hold_pips
        totals["cut_to_reserve"] += cut_pips
        totals["rotate_full_to_next"] += action_values["ROTATE_FULL_TO_NEXT"]
        totals["full_allocation_oracle"] += oracle_pips
        answer = {
            "id": packet["id"],
            "existing_pair": private["existing_pair"],
            "candidate_pair": private["candidate_pair"],
            "entry_epoch_utc": datetime.fromtimestamp(
                private["entry_epoch"], tz=UTC
            ).isoformat(),
            "decision_epoch_utc": datetime.fromtimestamp(
                private["decision_epoch"], tz=UTC
            ).isoformat(),
            "horizon_end_epoch_utc": datetime.fromtimestamp(
                private["decision_epoch"] + HOLD_HORIZON_S, tz=UTC
            ).isoformat(),
            "cut_pips": cut_pips,
            "hold_pips": hold_pips,
            "hold_incremental_vs_cut_pips": round(hold_pips - cut_pips, 2),
            "next_pips": next_pips,
            "existing_direction_truth": existing_truth,
            "next_direction_truth": next_truth,
        }
        answer_rows.append(answer)
        cells.append(
            {
                "id": packet["id"],
                "packet_canonical_sha256": packet["packet_canonical_sha256"],
                "decision": decision,
                "response_status": manifest_row["effective_status"],
                "fixed_policy_action": selected_action,
                "truth_revealed_after_response": answer,
                "hierarchical_policy_pips": hierarchical_pips,
                "full_hold_pips": hold_pips,
                "cut_to_reserve_pips": cut_pips,
                "rotate_full_to_next_pips": action_values["ROTATE_FULL_TO_NEXT"],
                "full_allocation_oracle_pips": oracle_pips,
                "full_allocation_oracle_action": oracle_action,
                "existing_direction_hit": existing_hit,
                "next_direction_hit": next_hit,
            }
        )

    answer_body = {
        "contract": "QR_DOJO_CAPITAL_RECYCLE_ANSWER_KEY_V2",
        "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "generated_after_response_manifest_raw_sha256": raw_sha(response_manifest_path),
        "source_corpus_inventory_canonical_sha256": corpus["canonical_sha256"],
        "decision_source_identities": [
            {
                "id": row["id"],
                "packet_canonical_sha256": row["packet_canonical_sha256"],
                "terminal_raw_sha256": row["terminal_raw_sha256"],
                "raw_response_raw_sha256": row["raw_response_raw_sha256"],
                "effective_status": row["effective_status"],
            }
            for row in manifest_rows
        ],
        "answers": answer_rows,
    }
    answer_key = {**answer_body, "canonical_sha256": canonical_sha(answer_body)}
    atomic_json(answer_path, answer_key)
    rounded_totals = {key: round(value, 3) for key, value in totals.items()}
    score_body = {
        "cell_count": CELL_COUNT,
        "totals_capacity_pips": rounded_totals,
        "delta_hierarchical_vs_full_hold_pips": round(
            totals["hierarchical"] - totals["full_hold"], 3
        ),
        "delta_hierarchical_vs_cut_pips": round(
            totals["hierarchical"] - totals["cut_to_reserve"], 3
        ),
        "delta_hierarchical_vs_rotate_pips": round(
            totals["hierarchical"] - totals["rotate_full_to_next"], 3
        ),
        "oracle_gap_pips": round(
            totals["hierarchical"] - totals["full_allocation_oracle"], 3
        ),
        "existing_direction_hits": existing_hits,
        "next_direction_hits": next_hits,
        "joint_direction_hits": joint_hits,
        "existing_direction_hit_rate": round(existing_hits / CELL_COUNT, 6),
        "next_direction_hit_rate": round(next_hits / CELL_COUNT, 6),
        "joint_direction_hit_rate": round(joint_hits / CELL_COUNT, 6),
        "absolute_profit_positive": totals["hierarchical"] > 0,
    }
    best_fixed = max(
        totals["full_hold"],
        totals["cut_to_reserve"],
        totals["rotate_full_to_next"],
    )
    score_body["verdict"] = (
        "POSITIVE_SMALL_WORN_TRAIN_HYPOTHESIS_NOT_PROOF"
        if totals["hierarchical"] > 0 and totals["hierarchical"] > best_fixed
        else "REJECTED_SMALL_WORN_TRAIN_HIERARCHICAL_POLICY"
    )
    evidence_body = {
        "contract": "QR_DOJO_AI_CAPITAL_RECYCLE_TRAIN_EVIDENCE_V2",
        "schema_version": 2,
        "experiment_id": EXPERIMENT_ID,
        "recorded_at_utc": datetime.now(tz=UTC).isoformat(),
        "classification": "SELF_ATTESTED_LOOKAHEAD_FREE_WORN_TRAIN_DIAGNOSTIC",
        "proof_eligible": False,
        "promotion_eligible": False,
        "live_permission": False,
        "order_authority": "NONE",
        "preregistration_raw_sha256": raw_sha(out / "preregistration.json"),
        "preregistration_canonical_sha256": prereg["canonical_sha256"],
        "packets_raw_sha256": raw_sha(out / "packets.json"),
        "prompt_template_raw_sha256": raw_sha(out / "prompt_template.txt"),
        "response_manifest_raw_sha256": raw_sha(response_manifest_path),
        "response_manifest_canonical_sha256": response_manifest["canonical_sha256"],
        "source_corpus_inventory_canonical_sha256": corpus["canonical_sha256"],
        "answer_key_raw_sha256": raw_sha(answer_path),
        "answer_key_canonical_sha256": answer_key["canonical_sha256"],
        "fresh_context_contract": {
            "one_terminal_per_context": True,
            "fixed_denominator": CELL_COUNT,
            "valid_context_count": sum(
                row["effective_status"] == "VALID" for row in manifest_rows
            ),
            "invalid_or_missing_counted_as_direction_miss": True,
            "tools_forbidden": True,
            "provider_context_nonreuse_externally_attested": False,
        },
        "fixed_policy": prereg_body["fixed_policy"],
        "cells": cells,
        "score": score_body,
        "limitations": [
            "2026_H1_HISTORY_ALREADY_WORN_BY_PRIOR_RESEARCH",
            "NO_GLOBAL_UNTOUCHED_HOLDOUT",
            "SIX_CELL_SMALL_SAMPLE",
            "MODEL_PROVIDER_IDENTITY_AND_CONTEXT_NONREUSE_NOT_EXTERNALLY_ATTESTED",
            "ADDITIONAL_SLIPPAGE_AND_FINANCING_NOT_CHARGED",
            "LINEAR_SINGLE_CAPACITY_MODEL_OMITS_MARGIN_AND_CURRENCY_CORRELATION",
            "RESULT_CANNOT_PROMOTE_OR_AUTHORIZE_LIVE_TRADING",
        ],
    }
    evidence = {**evidence_body, "canonical_sha256": canonical_sha(evidence_body)}
    atomic_json(result_path, evidence)
    print(json.dumps({"status": "SCORED", "score": score_body}, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-lineage", required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    seal_parser = sub.add_parser("seal-cell")
    seal_parser.add_argument("--cell-id", required=True)
    seal_parser.add_argument("--response", type=Path, required=True)
    missing_parser = sub.add_parser("seal-missing-cell")
    missing_parser.add_argument("--cell-id", required=True)
    sub.add_parser("finalize-responses")
    sub.add_parser("score")
    args = parser.parse_args()
    if args.command == "prepare":
        return prepare(args)
    if args.command == "seal-cell":
        return seal_cell(args)
    if args.command == "seal-missing-cell":
        return seal_missing_cell(args)
    if args.command == "finalize-responses":
        return finalize_responses(args)
    return score(args)


if __name__ == "__main__":
    raise SystemExit(main())
