"""Read-only OANDA S5 truth adapter for technical forward shadows."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from quant_rabbit.broker.oanda import OandaReadOnlyClient
from quant_rabbit.technical_forecast_forward_outcome import (
    S5BidAskCandle,
    append_forward_outcomes,
    build_forward_scorecard,
    load_forward_outcomes,
    load_forward_shadows,
    pending_forward_signals,
    resolve_frozen_forward_signal,
    write_json_atomic,
)
from quant_rabbit.technical_forecast_forward_shadow import load_forward_candidate


TRUTH_ADAPTER_CONTRACT = "QR_TECHNICAL_FORECAST_FORWARD_TRUTH_ADAPTER_V1"
MAX_ERROR_COUNT = 20
_OANDA_UTC_RE = re.compile(
    r"^(?P<seconds>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})"
    r"(?:\.(?P<fraction>\d{1,9}))?Z$"
)


class _ReadOnlyClient(Protocol):
    def get_json(
        self,
        path: str,
        query: dict[str, str] | None = None,
    ) -> dict[str, Any]: ...


def resolve_due_forward_outcomes_from_oanda(
    *,
    candidate_path: Path,
    shadow_ledger_path: Path,
    outcome_ledger_path: Path,
    scorecard_path: Path,
    client_factory: Callable[[], _ReadOnlyClient] = OandaReadOnlyClient,
    clock: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Resolve the bounded oldest due set; never call OANDA when none is due."""

    as_of = _aware_utc((clock or _utc_now)())
    base = {
        "contract": TRUTH_ADAPTER_CONTRACT,
        "schema_version": 1,
        "as_of_utc": as_of.isoformat(),
        "shadow_only": True,
        "broker_read": False,
        "broker_mutation": False,
        "live_order_enabled": False,
        "promotion_allowed": False,
        "order_intents": [],
    }
    try:
        candidate = load_forward_candidate(candidate_path)
        candidate_sha = _file_sha256(candidate_path)
        shadows = load_forward_shadows(
            shadow_ledger_path,
            candidate_sha256=candidate_sha,
        )
        outcomes = load_forward_outcomes(
            outcome_ledger_path,
            candidate_sha256=candidate_sha,
        )
        plan = pending_forward_signals(
            candidate,
            shadows,
            outcomes,
            as_of_utc=as_of,
        )
    except Exception as exc:
        return {
            **base,
            "status": "LEDGER_OR_CANDIDATE_INVALID",
            "output_unchanged": True,
            "errors": [_error("PRECHECK", exc)],
        }
    if not shadows:
        return {
            **base,
            "status": "NO_SHADOW_LEDGER",
            "output_unchanged": True,
            "emitted_signal_count": 0,
        }
    selected = plan["selected"]
    if not selected:
        return {
            **base,
            "status": "NO_DUE_SIGNALS",
            "output_unchanged": True,
            "emitted_signal_count": plan["emitted_signal_count"],
            "resolved_signal_count": plan["resolved_signal_count"],
            "pending_not_mature_count": plan["pending_not_mature_count"],
            "due_unresolved_count": plan["due_unresolved_count"],
        }

    try:
        client = client_factory()
    except Exception as exc:
        return {
            **base,
            "status": "CLIENT_UNAVAILABLE",
            "output_unchanged": True,
            "due_unresolved_count": plan["due_unresolved_count"],
            "errors": [_error("CLIENT", exc)],
        }
    workers = int(candidate["resolver"]["max_workers"])
    resolved_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    def resolve(task: Mapping[str, Any]) -> dict[str, Any]:
        candles, chunk_hashes = fetch_frozen_s5_truth(
            client,
            pair=str(task["pair"]),
            time_from=_parse_utc(task["decision_at_utc"]),
            time_to=_parse_utc(task["maturity_at_utc"]),
            chunk_candle_limit=int(candidate["resolver"]["chunk_candle_limit"]),
        )
        return resolve_frozen_forward_signal(
            candidate,
            task,
            candles,
            candidate_sha256=candidate_sha,
            resolved_at_utc=as_of,
            truth_chunk_sha256=chunk_hashes,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(resolve, task): task for task in selected}
        for future in concurrent.futures.as_completed(futures):
            task = futures[future]
            try:
                resolved_rows.append(future.result())
            except Exception as exc:
                errors.append(
                    _error(
                        "TRUTH_FETCH_OR_RESOLUTION",
                        exc,
                        pair=str(task.get("pair") or ""),
                        signal_sha256=str(task.get("signal_sha256") or ""),
                    )
                )
    errors = errors[:MAX_ERROR_COUNT]
    try:
        appended = append_forward_outcomes(
            outcome_ledger_path,
            resolved_rows,
            candidate_sha256=candidate_sha,
        )
        outcomes = load_forward_outcomes(
            outcome_ledger_path,
            candidate_sha256=candidate_sha,
        )
        scorecard = build_forward_scorecard(
            candidate,
            shadows,
            outcomes,
            candidate_sha256=candidate_sha,
            as_of_utc=as_of,
            acquisition_errors=errors,
        )
        write_json_atomic(scorecard_path, scorecard)
    except Exception as exc:
        return {
            **base,
            "status": "OUTCOME_PERSISTENCE_FAILED",
            "broker_read": True,
            "output_unchanged": True,
            "selected_due_count": len(selected),
            "resolved_in_memory_count": len(resolved_rows),
            "errors": [*errors, _error("PERSISTENCE", exc)][:MAX_ERROR_COUNT],
        }
    return {
        **base,
        "status": (
            "RESOLVED_WITH_ERRORS" if errors else "RESOLVED"
        ),
        "broker_read": True,
        "output_unchanged": False,
        "selected_due_count": len(selected),
        "resolved_in_memory_count": len(resolved_rows),
        "ledger_appended_count": appended,
        "deferred_due_count": plan["deferred_due_count"],
        "scorecard_status": scorecard["status"],
        "forward_evidence_passed": scorecard["forward_evidence_passed"],
        "promotion_allowed": False,
        "errors": errors,
    }


def fetch_frozen_s5_truth(
    client: _ReadOnlyClient,
    *,
    pair: str,
    time_from: datetime,
    time_to: datetime,
    chunk_candle_limit: int,
) -> tuple[list[S5BidAskCandle], list[str]]:
    """Fetch a complete fixed interval in bounded S5 BA requests."""

    start = _aware_utc(time_from)
    end = _aware_utc(time_to)
    if end <= start:
        raise ValueError("truth interval must be positive")
    if chunk_candle_limit.__class__ is not int or not 1 <= chunk_candle_limit <= 5000:
        raise ValueError("chunk_candle_limit must be an exact integer inside 1..5000")
    step = timedelta(seconds=5 * chunk_candle_limit)
    cursor = start
    chunk_hashes: list[str] = []
    by_timestamp: dict[datetime, S5BidAskCandle] = {}
    while cursor < end:
        chunk_end = min(end, cursor + step)
        payload = client.get_json(
            f"/v3/instruments/{pair}/candles",
            {
                "granularity": "S5",
                "from": _format_oanda_time(cursor),
                "to": _format_oanda_time(chunk_end),
                "price": "BA",
                "includeFirst": "true",
                "smooth": "false",
            },
        )
        if not isinstance(payload, dict) or not isinstance(payload.get("candles"), list):
            raise ValueError("OANDA S5 response is not a candle object")
        if payload.get("instrument") != pair or payload.get("granularity") != "S5":
            raise ValueError("OANDA S5 response provenance does not match the request")
        if len(payload["candles"]) > chunk_candle_limit:
            raise ValueError("OANDA S5 response exceeds the requested chunk bound")
        chunk_hashes.append(_stable_digest(payload))
        for item in payload["candles"]:
            candle = _parse_oanda_candle(item, pair=pair)
            if not cursor <= candle.timestamp_utc < chunk_end:
                raise ValueError("OANDA S5 candle lies outside its requested chunk")
            previous = by_timestamp.get(candle.timestamp_utc)
            if previous is not None and previous != candle:
                raise ValueError("OANDA S5 duplicate timestamp has conflicting prices")
            by_timestamp[candle.timestamp_utc] = candle
        cursor = chunk_end
    candles = [by_timestamp[timestamp] for timestamp in sorted(by_timestamp)]
    return candles, chunk_hashes


def _parse_oanda_candle(value: Any, *, pair: str) -> S5BidAskCandle:
    if not isinstance(value, Mapping):
        raise ValueError(f"{pair}: OANDA S5 candle is not an object")
    if value.get("complete") is not True:
        raise ValueError(f"{pair}: OANDA S5 truth contains an incomplete candle")
    timestamp = _parse_oanda_utc(value.get("time"))
    bid = value.get("bid")
    ask = value.get("ask")
    if not isinstance(bid, Mapping) or not isinstance(ask, Mapping):
        raise ValueError(f"{pair}: OANDA S5 bid/ask blocks are missing")
    prices: dict[str, float] = {}
    for prefix, block in (("bid", bid), ("ask", ask)):
        for short, name in (("o", "o"), ("h", "h"), ("l", "l"), ("c", "c")):
            parsed = _finite(block.get(short))
            if parsed is None or parsed <= 0.0:
                raise ValueError(f"{pair}: OANDA S5 {prefix}.{short} is invalid")
            prices[f"{prefix}_{name}"] = parsed
    return S5BidAskCandle(timestamp_utc=timestamp, **prices)


def _parse_oanda_utc(value: Any) -> datetime:
    match = _OANDA_UTC_RE.fullmatch(str(value or ""))
    if match is None:
        raise ValueError("OANDA S5 timestamp is not canonical UTC")
    fraction = (match.group("fraction") or "")[:6].ljust(6, "0")
    parsed = datetime.fromisoformat(
        match.group("seconds") + (f".{fraction}" if fraction else "") + "+00:00"
    )
    return parsed.astimezone(timezone.utc)


def _parse_utc(value: Any) -> datetime:
    text = str(value or "").strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError("timestamp must be aware UTC") from exc
    return _aware_utc(parsed)


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc)


def _format_oanda_time(value: datetime) -> str:
    return _aware_utc(value).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _finite(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def _stable_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _error(code: str, exc: Exception, **context: str) -> dict[str, Any]:
    return {
        "code": code,
        "message": f"{exc.__class__.__name__}: {exc}"[:320],
        **context,
    }


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


__all__ = [
    "TRUTH_ADAPTER_CONTRACT",
    "fetch_frozen_s5_truth",
    "resolve_due_forward_outcomes_from_oanda",
]
