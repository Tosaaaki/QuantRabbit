"""Exact OANDA S5 bid/ask outcomes for deterministic fast-bot shadows."""

from __future__ import annotations

import concurrent.futures
import fcntl
import hashlib
import json
import math
import os
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from quant_rabbit.broker.oanda import OandaReadOnlyClient
from quant_rabbit.fast_bot import METHODS, SIGNAL_CONTRACT
from quant_rabbit.technical_forecast_forward_outcome import S5BidAskCandle
from quant_rabbit.technical_forecast_forward_truth import fetch_frozen_s5_truth


OUTCOME_CONTRACT = "QR_FAST_BOT_S5_BID_ASK_OUTCOME_V1"
SCORECARD_CONTRACT = "QR_FAST_BOT_FORWARD_SCORECARD_V1"
TRUTH_ADAPTER_CONTRACT = "QR_FAST_BOT_S5_TRUTH_ADAPTER_V1"
MAX_DUE_PER_RUN = 12
MAX_WORKERS = 4


def resolve_fast_bot_signal(
    signal: Mapping[str, Any],
    candles: Sequence[S5BidAskCandle],
    *,
    resolved_at_utc: datetime,
    truth_chunk_sha256: Sequence[str] = (),
) -> dict[str, Any]:
    """Resolve one passive LIMIT signal using executable bid/ask sides.

    No-fill is zero.  A filled signal that has not reached TP before the fixed
    horizon is conservatively charged the full attached SL.  Same-S5 TP/SL is
    stop-first.  This scorer therefore cannot manufacture edge through a
    favorable intrabar or time-close assumption.
    """

    _validate_signal(signal)
    resolved = _aware_utc(resolved_at_utc)
    generated = _parse_utc(signal["generated_at_utc"])
    side = str(signal["side"])
    entry = float(signal["entry"])
    tp = float(signal["take_profit"])
    sl = float(signal["stop_loss"])
    tp_pips = float(signal["take_profit_pips"])
    sl_pips = float(signal["stop_loss_pips"])
    ttl = int(signal["entry_ttl_seconds"])
    hold = int(signal["max_hold_seconds"])
    fill_deadline = generated + timedelta(seconds=ttl)
    maturity = fill_deadline + timedelta(seconds=hold)
    if resolved < maturity:
        raise ValueError("fast-bot signal is not mature")
    ordered = sorted(
        (item for item in candles if generated <= item.timestamp_utc < maturity),
        key=lambda item: item.timestamp_utc,
    )
    if (
        not ordered
        or ordered[0].timestamp_utc > generated + timedelta(seconds=10)
        or (
            ordered[-1].timestamp_utc + timedelta(seconds=5)
            < maturity - timedelta(seconds=10)
        )
    ):
        raise ValueError("incomplete fast-bot S5 truth coverage")
    fill_at: datetime | None = None
    exit_at: datetime | None = None
    exit_reason = "UNFILLED"
    realized_pips = 0.0
    ambiguous = False
    observed = 0
    for candle in ordered:
        observed += 1
        if fill_at is None:
            filled = (
                candle.ask_l <= entry
                if side == "LONG"
                else candle.bid_h >= entry
            )
            if not filled or candle.timestamp_utc > fill_deadline:
                continue
            fill_at = candle.timestamp_utc
        if candle.timestamp_utc >= fill_at + timedelta(seconds=hold):
            break
        if side == "LONG":
            tp_hit = candle.bid_h >= tp
            sl_hit = candle.bid_l <= sl
        else:
            tp_hit = candle.ask_l <= tp
            sl_hit = candle.ask_h >= sl
        if tp_hit and sl_hit:
            ambiguous = True
            exit_reason = "STOP_LOSS_AMBIGUOUS_SAME_S5"
            realized_pips = -sl_pips
            exit_at = candle.timestamp_utc
            break
        if sl_hit:
            exit_reason = "STOP_LOSS"
            realized_pips = -sl_pips
            exit_at = candle.timestamp_utc
            break
        if tp_hit:
            exit_reason = "TAKE_PROFIT"
            realized_pips = tp_pips
            exit_at = candle.timestamp_utc
            break
    if fill_at is not None and exit_at is None:
        exit_reason = "HORIZON_FULL_STOP_LOSS"
        realized_pips = -sl_pips
        exit_at = min(fill_at + timedelta(seconds=hold), resolved)
    body = {
        "contract": OUTCOME_CONTRACT,
        "schema_version": 1,
        "signal_id": str(signal["signal_id"]),
        "pair": str(signal["pair"]),
        "side": side,
        "method": str(signal["method"]),
        "signal_generated_at_utc": generated.isoformat(),
        "resolved_at_utc": resolved.isoformat(),
        "maturity_at_utc": maturity.isoformat(),
        "filled": fill_at is not None,
        "fill_at_utc": fill_at.isoformat() if fill_at else None,
        "exit_at_utc": exit_at.isoformat() if exit_at else None,
        "exit_reason": exit_reason,
        "realized_pips": round(realized_pips, 6),
        "ambiguous_same_s5": ambiguous,
        "truth_source": "OANDA_S5_BID_ASK",
        "truth_candle_count": observed,
        "truth_chunk_sha256": [str(item) for item in truth_chunk_sha256],
        "signal_sha256": str(signal["signal_sha256"]),
        "shadow_only": True,
        "live_permission": False,
        "broker_mutation": False,
    }
    return _seal(body)


def build_fast_bot_scorecard(
    signals: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
    *,
    as_of_utc: datetime,
) -> dict[str, Any]:
    valid_signal_rows = [item for item in signals if _fast_bot_signal_valid(item)]
    valid_signals = _dedupe_signal_identities(valid_signal_rows)
    valid_outcomes = [
        item
        for item in outcomes
        if _sealed_valid(item, OUTCOME_CONTRACT)
    ]
    by_signal = {str(item["signal_sha256"]): item for item in valid_outcomes}
    emitted = valid_signals
    resolved = [
        by_signal[str(item["signal_sha256"])]
        for item in emitted
        if str(item["signal_sha256"]) in by_signal
        and by_signal[str(item["signal_sha256"])].get("signal_id") == item.get("signal_id")
    ]
    fills = [item for item in resolved if item.get("filled") is True]
    values = [float(item["realized_pips"]) for item in fills]
    wins = [value for value in values if value > 0.0]
    losses = [value for value in values if value < 0.0]
    active_days = {
        _parse_utc(item["generated_at_utc"]).date().isoformat()
        for item in emitted
        if isinstance(item.get("generated_at_utc"), str)
    }
    mean = statistics.fmean(values) if values else None
    lower = _one_sided_95_mean_lower(values)
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0.0
        else math.inf if gross_profit > 0.0 else None
    )
    passed = bool(
        len(fills) >= 100
        and len(active_days) >= 10
        and mean is not None
        and mean > 0.0
        and lower is not None
        and lower > 0.0
        and profit_factor is not None
        and profit_factor >= 1.25
    )
    body = {
        "contract": SCORECARD_CONTRACT,
        "schema_version": 1,
        "generated_at_utc": _aware_utc(as_of_utc).isoformat(),
        "status": "FORWARD_EVIDENCE_PASSED" if passed else "COLLECTING_FORWARD_EVIDENCE",
        "emitted_signals": len(emitted),
        "duplicate_identity_signals_ignored": len(valid_signal_rows) - len(valid_signals),
        "resolved_signals": len(resolved),
        "filled_signals": len(fills),
        "unfilled_signals": sum(item.get("filled") is False for item in resolved),
        "active_days": len(active_days),
        "wins": len(wins),
        "losses": len(losses),
        "fill_rate": round(len(fills) / len(resolved), 6) if resolved else None,
        "win_rate": round(len(wins) / len(fills), 6) if fills else None,
        "net_pips": round(sum(values), 6),
        "mean_pips_per_fill": round(mean, 6) if mean is not None else None,
        "one_sided_95_mean_lower_pips": (
            round(lower, 6)
            if lower is not None and math.isfinite(lower)
            else None
        ),
        "profit_factor": round(profit_factor, 6) if profit_factor is not None and math.isfinite(profit_factor) else "INF" if profit_factor == math.inf else None,
        "forward_evidence_passed": passed,
        "live_permission": False,
        "promotion_allowed": False,
        "promotion_blockers": (
            ["SEPARATE_CONTENT_ADDRESSED_LIVE_PROMOTION_REQUIRED"]
            if passed
            else [
                "MINIMUM_100_EXACT_S5_FILLS_NOT_MET" if len(fills) < 100 else None,
                "MINIMUM_10_ACTIVE_DAYS_NOT_MET" if len(active_days) < 10 else None,
                "POST_COST_EXPECTANCY_LOWER_BOUND_NOT_POSITIVE" if lower is None or lower <= 0.0 else None,
                "PROFIT_FACTOR_1_25_NOT_MET" if profit_factor is None or profit_factor < 1.25 else None,
                "SEPARATE_CONTENT_ADDRESSED_LIVE_PROMOTION_REQUIRED",
            ]
        ),
        "shadow_only": True,
        "broker_mutation": False,
    }
    body["promotion_blockers"] = [item for item in body["promotion_blockers"] if item]
    return _seal(body)


def resolve_due_fast_bot_outcomes_from_oanda(
    *,
    shadow_ledger_path: Path,
    outcome_ledger_path: Path,
    scorecard_path: Path,
    client_factory: Callable[[], Any] = OandaReadOnlyClient,
    clock: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    now = _aware_utc((clock or (lambda: datetime.now(timezone.utc)))())
    loaded_signals = _load_jsonl(shadow_ledger_path)
    valid_loaded_signals = [
        item for item in loaded_signals if _fast_bot_signal_valid(item)
    ]
    signals = _dedupe_signal_identities(valid_loaded_signals)
    outcomes = _load_jsonl(outcome_ledger_path)
    resolved_ids = {
        str(item.get("signal_id"))
        for item in outcomes
        if _sealed_valid(item, OUTCOME_CONTRACT)
    }
    due = []
    for signal in signals:
        if not _fast_bot_signal_valid(signal) or str(signal.get("signal_id")) in resolved_ids:
            continue
        generated = _parse_utc(signal["generated_at_utc"])
        maturity = generated + timedelta(
            seconds=int(signal["entry_ttl_seconds"]) + int(signal["max_hold_seconds"])
        )
        if maturity <= now:
            due.append((maturity, signal))
    due.sort(key=lambda item: (item[0], str(item[1].get("signal_id"))))
    selected = [item[1] for item in due[:MAX_DUE_PER_RUN]]
    base = {
        "contract": TRUTH_ADAPTER_CONTRACT,
        "schema_version": 1,
        "generated_at_utc": now.isoformat(),
        "shadow_only": True,
        "live_permission": False,
        "broker_mutation": False,
        "due_count": len(due),
        "selected_due_count": len(selected),
        "duplicate_identity_signals_ignored": (
            len(valid_loaded_signals) - len(signals)
        ),
    }
    if not selected:
        scorecard = build_fast_bot_scorecard(loaded_signals, outcomes, as_of_utc=now)
        _write_json_atomic(scorecard_path, scorecard)
        return {**base, "status": "NO_DUE_SIGNALS", "broker_read": False, "ledger_appended": 0, "scorecard_status": scorecard["status"]}

    client = client_factory()
    resolved: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    def resolve(signal: Mapping[str, Any]) -> dict[str, Any]:
        generated = _parse_utc(signal["generated_at_utc"])
        maturity = generated + timedelta(
            seconds=int(signal["entry_ttl_seconds"]) + int(signal["max_hold_seconds"])
        )
        candles, hashes = fetch_frozen_s5_truth(
            client,
            pair=str(signal["pair"]),
            time_from=generated,
            time_to=maturity,
            chunk_candle_limit=4500,
        )
        return resolve_fast_bot_signal(signal, candles, resolved_at_utc=now, truth_chunk_sha256=hashes)

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(resolve, signal): signal for signal in selected}
        for future in concurrent.futures.as_completed(futures):
            signal = futures[future]
            try:
                resolved.append(future.result())
            except Exception as exc:  # pragma: no cover - network boundary
                errors.append({
                    "signal_id": str(signal.get("signal_id") or ""),
                    "pair": str(signal.get("pair") or ""),
                    "error": f"{type(exc).__name__}: {exc}"[:320],
                })
    appended = _append_outcomes_once(outcome_ledger_path, resolved)
    all_outcomes = _load_jsonl(outcome_ledger_path)
    scorecard = build_fast_bot_scorecard(
        loaded_signals,
        all_outcomes,
        as_of_utc=now,
    )
    _write_json_atomic(scorecard_path, scorecard)
    return {
        **base,
        "status": "RESOLVED_WITH_ERRORS" if errors else "RESOLVED",
        "broker_read": True,
        "ledger_appended": appended,
        "errors": errors,
        "scorecard_status": scorecard["status"],
        "forward_evidence_passed": scorecard["forward_evidence_passed"],
    }


def _validate_signal(signal: Mapping[str, Any]) -> None:
    if not _fast_bot_signal_valid(signal):
        raise ValueError("invalid fast-bot signal")


def _fast_bot_signal_valid(signal: Mapping[str, Any]) -> bool:
    try:
        signal_body = {key: item for key, item in signal.items() if key != "signal_sha256"}
        signal_sha = str(signal["signal_sha256"])
        regime_sha = str(signal["regime_contract_sha256"])
        signal_id = str(signal["signal_id"])
        pair = str(signal["pair"])
        side = str(signal.get("side") or "")
        method = str(signal["method"])
        entry = float(signal["entry"])
        tp = float(signal["take_profit"])
        sl = float(signal["stop_loss"])
        tp_pips = float(signal["take_profit_pips"])
        sl_pips = float(signal["stop_loss_pips"])
        reward_risk = float(signal["reward_risk"])
        generated = _parse_utc(signal["generated_at_utc"])
        quote_at = _parse_utc(signal["quote_timestamp_utc"])
        m1_closed = _parse_utc(signal["m1_closed_candle_utc"])
        ttl = int(signal["entry_ttl_seconds"])
        hold = int(signal["max_hold_seconds"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    geometry_ok = (
        tp > entry > sl if side == "LONG" else sl > entry > tp if side == "SHORT" else False
    )
    digest_ok = (
        _sha256_text(signal_sha)
        and _sha256_text(regime_sha)
        and signal_sha == _canonical_sha(signal_body)
    )
    timing_ok = (
        abs((generated - quote_at).total_seconds()) <= 45.0
        and 0.0 <= (generated - m1_closed).total_seconds() <= 120.0
    )
    return bool(
        signal.get("contract") == SIGNAL_CONTRACT
        and signal.get("schema_version") == 1
        and digest_ok
        and len(signal_id) == 24
        and all(character in "0123456789abcdef" for character in signal_id)
        and pair == pair.upper()
        and "_" in pair
        and method in METHODS
        and signal.get("shadow_only") is True
        and signal.get("live_permission") is False
        and signal.get("broker_mutation_allowed") is False
        and str(signal.get("order_type") or "") == "LIMIT"
        and str(signal.get("entry_reference") or "") == "PASSIVE_NEAR_SIDE"
        and signal.get("attached_take_profit_required") is True
        and signal.get("attached_stop_loss_required") is True
        and geometry_ok
        and timing_ok
        and tp_pips > 0.0
        and sl_pips > 0.0
        and math.isclose(
            reward_risk,
            tp_pips / sl_pips,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        and ttl == 90
        and hold == 15 * 60
    )


def _one_sided_95_mean_lower(values: Sequence[float]) -> float | None:
    if not values:
        return None
    mean = statistics.fmean(values)
    if len(values) == 1:
        return -math.inf
    stdev = statistics.stdev(values)
    if stdev == 0.0:
        return mean
    critical = _student_t_one_sided_95(len(values) - 1)
    return mean - critical * stdev / math.sqrt(len(values))


def _dedupe_signal_identities(
    signals: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Keep the first immutable hypothesis for each pair/closed-M1 seat.

    Revision-1 signals briefly included the whole regime-contract digest in
    ``signal_id``.  A quote-only 30-second refresh could therefore append the
    same M1 seat twice.  The append-only ledger is preserved, while both due
    resolution and scorecards ignore every later duplicate identity.
    """

    seen: set[tuple[str, str]] = set()
    out: list[Mapping[str, Any]] = []
    for signal in signals:
        identity = (
            str(signal.get("pair") or ""),
            str(signal.get("m1_closed_candle_utc") or ""),
        )
        if identity in seen:
            continue
        seen.add(identity)
        out.append(signal)
    return out


def _student_t_one_sided_95(df: int) -> float:
    table = {
        1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015,
        6: 1.943, 7: 1.895, 8: 1.860, 9: 1.833, 10: 1.812,
        12: 1.782, 15: 1.753, 20: 1.725, 25: 1.708, 30: 1.697,
        40: 1.684, 60: 1.671, 120: 1.658,
    }
    for bound in sorted(table):
        if df <= bound:
            return table[bound]
    return 1.645


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    except (OSError, json.JSONDecodeError, ValueError):
        return []
    return rows


def _append_outcomes_once(path: Path, outcomes: Sequence[Mapping[str, Any]]) -> int:
    if not outcomes:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        seen = set()
        for line in handle:
            try:
                item = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(item, Mapping) and item.get("signal_id"):
                seen.add(str(item["signal_id"]))
        handle.seek(0, os.SEEK_END)
        appended = 0
        for outcome in outcomes:
            signal_id = str(outcome.get("signal_id") or "")
            if not signal_id or signal_id in seen or not _sealed_valid(outcome, OUTCOME_CONTRACT):
                continue
            handle.write(json.dumps(dict(outcome), ensure_ascii=False, sort_keys=True) + "\n")
            seen.add(signal_id)
            appended += 1
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return appended


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    body = {key: item for key, item in value.items() if key != "contract_sha256"}
    return {**body, "contract_sha256": _canonical_sha(body)}


def _sealed_valid(value: Mapping[str, Any], contract: str) -> bool:
    if not isinstance(value, Mapping) or value.get("contract") != contract:
        return False
    stored = str(value.get("contract_sha256") or "")
    body = {key: item for key, item in value.items() if key != "contract_sha256"}
    return bool(stored and stored == _canonical_sha(body))


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _sha256_text(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _parse_utc(value: Any) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("timestamp must be aware UTC") from exc
    return _aware_utc(parsed)


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc)


__all__ = [
    "OUTCOME_CONTRACT",
    "SCORECARD_CONTRACT",
    "build_fast_bot_scorecard",
    "resolve_due_fast_bot_outcomes_from_oanda",
    "resolve_fast_bot_signal",
]
